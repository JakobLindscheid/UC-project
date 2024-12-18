import numpy as np
import duckdb
import pandas as pd
import shapely
from tqdm import tqdm

from osgeo import gdal
import geopandas as gpd
gdal.SetConfigOption('SHAPE_RESTORE_SHX', 'YES')

CATEGORIES = "lvl2"
SQUARE_SIZE = 1000
CHECK_INS = False
NEIGHBORS = False

if CATEGORIES == "handpicked":

    categories = {
        # 6/12 mentioned in the paper
        "education": (2, 'Education'),
        "scenic": (1, "Landmarks and Outdoors"),
        "sports": (1,"Sports and Recreation"),
        "commercial": (1,"Retail"),
        "financial": (2,"Financial Service"),
        "transport": (2,"Transport Hub"),
        # added by us
        "entertainment": (1, "Arts and Entertainment"),
        "office": (2, "Office"),
        "government": (2, "Government Building"),
        "food": (2, "Restaurant"),
        "health": (1, "Health and Medicine"),
        "hotel": (2, "Lodging")
    }

elif CATEGORIES == "lvl2":
    categories = {
        c.lower().replace(" ","_"): (2, c)
        for c in pd.read_csv("foursquare_categories.csv")["category_label"].str.split(" > ", expand=True)[1].unique()
        if c is not None
    }
print(len(categories.keys()), " categories")

cities = {
    "Shanghai": {
        "city_names": ["Shanghai", "上海"],
        "shape_file": 'shanghai-provjson.shp',
    },
    "Nanjing": {
        "city_names": ["Nanjing", "南京"],
        "corner1": (118.39246295229297, 31.261649948659116),
        "corner2": (119.22570190410435, 32.56306709606652)
    },
    "Beijing": {
        "city_names": ["Beijing", "北京"],
        "corner1": (115.39783664651948, 39.45697643649945),
        "corner2": (117.59107359664718, 41.053361082863574), 
    },
    "Xi'An": {
        "city_names": ["Xi''an", "西安"],
        "corner1": (107.58696095508553, 33.684721810688416),
        "corner2": (109.828485064734, 34.833361874606794), 
    }
}
print(len(cities.keys()), " cities")

r_earth = 6_371_000 # meters
def move_lat(lat, d_lat):
    lat += d_lat / r_earth * 180 / np.pi
    return lat
def move_long(long, d_long, lat):
    long += d_long / r_earth * 180 / np.pi / np.cos(lat * np.pi / 180)
    return long


def generate_squares(min_lat, max_lat, min_long, max_long, square_size=500):
    long = min_long
    lat = min_lat

    columns={"index1":[], "index2":[], "min_long":[], "min_lat":[], "max_long":[], "max_lat":[]}

    i = 0
    while lat <= max_lat:
        next_lat = move_lat(lat, square_size)

        j = 0
        while long <= max_long:
            next_long = move_long(long, square_size, lat)

            columns["index1"].append(i)
            columns["index2"].append(j)
            columns["min_long"].append(long)
            columns["min_lat"].append(lat)
            columns["max_long"].append(next_long)
            columns["max_lat"].append(next_lat)

            long = next_long
            j += 1
            
        i += 1
        lat = next_lat
        long = min_long    

    squares = pd.DataFrame(columns)
    return squares


def query_pois(con, boundaries, categories, city_names=["Shanghai", "上海"]):
    min_lat, max_lat, min_long, max_long = boundaries
    cat_condition = " OR ".join([f"""level{level}_category_name = '{name.replace("'", "''")}'""" for level, name in categories.values()])
    df = con.sql(f"""
            WITH filtered_places AS (
                SELECT * FROM places WHERE ({" OR ".join(["'" + city_name + "' IN locality" for city_name in city_names])})
            ), 
            exploded_categories AS (
                SELECT DISTINCT fsq_place_id
                FROM (
                    SELECT fsq_place_id,
                        UNNEST(fsq_category_ids) as fsq_category_id
                    FROM filtered_places
                ) as p 
                JOIN categories AS c 
                ON p.fsq_category_id = c.category_id
                
                WHERE {cat_condition}
            )
            SELECT * FROM places as p JOIN exploded_categories as c ON p.fsq_place_id = c.fsq_place_id
            WHERE latitude <= {max_lat} AND latitude >= {min_lat} AND longitude <= {max_long} AND longitude >= {min_long};
            """
    ).df()
    return df


def poi_counts(con,  squares, categories, city_names, boundaries):
    
    df = query_pois(con, boundaries, categories, city_names)
        
    df["square"] = df.apply(lambda x:
        squares[
            (squares["min_long"]<= x["longitude"]) & (squares["max_long"]>x["longitude"]) &
            (squares["min_lat"]<= x["latitude"]) & (squares["max_lat"]>x["latitude"])
        ].index[0], 
    axis=1)
    
    df = df.assign(category="")
    for cat in tqdm(categories.keys(), desc=city_names[0]):
        df.loc[
            df[df["fsq_category_labels"].astype(str).str.contains(categories[cat][1])].index,
            "category"
        ] = cat
    

    return df

if __name__ == "__main__":

    con = duckdb.connect("C:/Users/Public/fsq/fsq.db")
    con.sql(
        "CREATE TABLE IF NOT EXISTS places AS SELECT * FROM read_parquet( 'C:/Users/Public/fsq/places-*.snappy.parquet');"
    )
    con.sql(
        "create TABLE IF NOT EXISTS categories as SELECT * FROM read_parquet( 'C:/Users/Public/fsq/categories.snappy.parquet');"
    )

    full_dataset = []
    for city in cities.keys():

        # get the boundaries of the city
        shape_file = cities[city].get("shape_file")
        if shape_file:
            geo_df = gpd.read_file(shape_file)

            polygons = geo_df.geometry.tolist()
            boundary = gpd.GeoSeries(shapely.ops.unary_union(polygons))
            long, lat = boundary.at[0].exterior.coords.xy

            min_long = min(long)
            max_long = max(long)
            min_lat = min(lat)
            max_lat = max(lat)

        else:
            corner1 = cities[city].get("corner1")
            corner2 = cities[city].get("corner2")
            if corner1 is None or corner2 is None:
                raise ValueError("Please provide either a shape file or the top left and bottom right coordinates.")
            
            min_long, min_lat = corner1
            max_long, max_lat = corner2
        
        # generate the squares
        squares = generate_squares(min_lat, max_lat, min_long, max_long, SQUARE_SIZE)

        # get the poi counts
        data = poi_counts(con, squares, categories, cities[city]["city_names"], (min_lat, max_lat, min_long, max_long))
        data = data.drop_duplicates(subset=["latitude", "longitude", "name"])
        
        # merge the data with the squares
        squares = pd.merge(
            squares, 
            data.pivot_table(index="square", columns="category", values="name", aggfunc="count", fill_value=0), 
            left_index=True, 
            right_index=True, 
            how="outer", 
        ).fillna(0)

        squares = squares.assign(label=city)
        full_dataset.append(squares)
    full_dataset = pd.concat(full_dataset, ignore_index=True)


    full_dataset = full_dataset.fillna(0)


    full_dataset = full_dataset.loc[:, [c for c in full_dataset.columns if c != "label"] + ["label"]]


    full_dataset = full_dataset[(full_dataset[full_dataset.columns[6:-1]]!=0).any(axis=1)].reset_index(drop=True)


    if CHECK_INS:
        checkin_poi = pd.read_csv(r"C:\Users\Public\LSBN2vec++_dataset_WWW2019\dataset_WWW2019\raw_POIs.txt", delimiter="\t", header=None)
        checkin_poi.columns = ["id", "latitude", "longitude", "category", "country"]


        checkin_poi = checkin_poi[checkin_poi["country"]=="CN"].reset_index(drop=True)
        checkin_poi["square"] = checkin_poi.apply(lambda x:
            full_dataset[
                (full_dataset["min_long"]<= x["longitude"]) & (full_dataset["max_long"]>x["longitude"]) &
                (full_dataset["min_lat"]<= x["latitude"]) & (full_dataset["max_lat"]>x["latitude"])
            ].index, 
        axis=1)
        checkin_poi["square"] = checkin_poi.apply(lambda x: x["square"][0] if len(x["square"]) > 0 else np.nan, axis=1)
        checkin_poi = checkin_poi.dropna(subset=["square"])


        checkins = pd.read_csv(r"C:\Users\Public\LSBN2vec++_dataset_WWW2019\dataset_WWW2019\raw_Checkins_anonymized.txt", delimiter="\t", header=None)
        checkins.columns = ["user", "poi", "datetime", "tz"]


        checkins = checkins[checkins["poi"].isin(checkin_poi["id"])]
        checkins = pd.merge(checkins, checkin_poi, left_on="poi", right_on="id")
        full_dataset = pd.merge(
            full_dataset, 
            checkins.groupby("square")["user"].count(), 
            left_index=True, 
            right_index=True,
            how="left"
        ).rename(columns={"user":"checkins"})
        full_dataset["checkins"] = full_dataset["checkins"].fillna(0)


    if NEIGHBORS:
        neighbors = [
            (0, 1),
            (1, 0),
            (0, -1),
            (-1, 0),
            (1, 1),
            (-1, -1),
            (1, -1),
            (-1, 1)
        ]


        neighbor_datasets = []
        for n1, n2 in neighbors:
            neighbor_idx = full_dataset.apply(lambda x: full_dataset[
                (full_dataset["label"] == x["label"]) &
                (full_dataset["index1"] == x["index1"] + n1) & 
                (full_dataset["index2"] == x["index2"] + n2)
            ].index, axis=1)
            neighbor_idx = neighbor_idx.apply(lambda x: x[0] if len(x) > 0 else np.nan)
            neighbor_idx = neighbor_idx[neighbor_idx.notna()]

            neighbor_data = full_dataset.loc[neighbor_idx, [c for c in full_dataset.columns[6:] if c != "label"]]
            neighbor_data.columns = [c + f"_n{n1},{n2}" for c in neighbor_data.columns]

            neighbor_datasets.append(neighbor_data)
        full_dataset = pd.concat([full_dataset]+neighbor_datasets, axis=1).fillna(0)


    full_dataset.to_csv(f"data/squares_{CATEGORIES}_cats_{SQUARE_SIZE}m{'_neighbors' if NEIGHBORS else ''}{'_checkins' if CHECK_INS else ''}.csv", index=False)
