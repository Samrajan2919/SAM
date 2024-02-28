from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import pandas as pd
from geopy.distance import geodesic
import folium
from folium.plugins import MarkerCluster


spark_config = {
    "spark.executor.memory": "8g",
    "spark.driver.memory": "4g",
    "spark.sql.shuffle.partitions": "200"
}

# Create a SparkSession
spark = SparkSession.builder \
    .appName('EarthquakeAnalysis') \
    .master("local[4]") \
    .config("spark.executor.memory", spark_config["spark.executor.memory"]) \
    .config("spark.driver.memory", spark_config["spark.driver.memory"]) \
    .config("spark.sql.shuffle.partitions", spark_config["spark.sql.shuffle.partitions"]) \
    .getOrCreate()


# Read the CSV file into a PySpark DataFrame
df = spark.read.csv("/location/database.csv", header=True)
df.show(1)

#df =df.limit(10)

# Check for null values in the DataFrame
null_values = df.select([col(c).alias(c.replace(" ", "_")) for c in df.columns]).na.drop().count()

print("Null Values in the DataFrame:")
print(null_values)

# Convert 'Date' column to timestamp type
df = df.withColumn("Date", to_timestamp("Date", "yyyy-MM-dd HH:mm:ss"))
df.show(2)

# Clean up 'Time' column (assuming 'Time' contains only time information)
df = df.withColumn("Time", to_timestamp("Time", "HH:mm:ss").cast("string"))
df.show(3)
# Combine 'Date' and 'Time' columns into a single 'Timestamp' column
df = df.withColumn("Timestamp", expr("unix_timestamp(Date) + unix_timestamp(Time)"))

df.show(4)
# Drop the original 'Date' and 'Time' columns if needed
df = df.drop("Date", "Time")

# Check for null values in the DataFrame after processing
null_values_processed = df.select([col(c).alias(c.replace(" ", "_")) for c in df.columns]).na.drop().count()

print("\nNull Values in the DataFrame after processing:")
print(null_values_processed)

# Filter the dataset to include only earthquakes with magnitude greater than 5.0
filtered_df = df.filter(col("Magnitude") > 5.0)

# Show the filtered DataFrame
filtered_df.show(5)


# Group by 'Type' and calculate the average depth and magnitude
average_stats_by_type = (
    filtered_df
    .groupBy('Type')
    .agg(mean('Depth').alias('Average_Depth'), mean('Magnitude').alias('Average_Magnitude'))
)

# Show the resulting DataFrame
average_stats_by_type.show(6)



# Define a UDF to categorize earthquake levels based on magnitude
# Define a UDF to categorize earthquake levels based on magnitude
def categorize_magnitude_level(magnitude):
    return when(magnitude < 6.0, 'Low').when((magnitude >= 6.0) & (magnitude < 7.0), 'Moderate').otherwise('High')

# Apply the UDF to create a new column 'Magnitude Level'
filtered_df = filtered_df.withColumn("Magnitude_Level", categorize_magnitude_level(col("Magnitude")))

# Show the DataFrame with the new column
filtered_df.show(7)


# Define the reference location
reference_latitude = 0
reference_longitude = 0

# User-Defined Function (UDF) to calculate geodesic distance and get location name
@udf(StringType())
def calculate_distance_and_location_udf(lat, lon):
    try:
        geolocator = Nominatim(user_agent="location_app")
        location = geolocator.reverse((lat, lon), language='en',timeout=20)
        location_name = location.address if location else "Location not found"
    except GeocoderTimedOut:
        location_name = "Location not found"
    
    return location_name

# User-Defined Function (UDF) to calculate geodesic distance
@udf(DoubleType())
def calculate_distance_udf(lat, lon):
    return float(geodesic((lat, lon), (reference_latitude, reference_longitude)).kilometers)

# Calculate the distance and get location name using the UDFs
filtered_df = filtered_df.withColumn(
    "Distance",
    calculate_distance_udf(col("Latitude"), col("Longitude"))
)
filtered_df = filtered_df.withColumn(
    "Location",
    calculate_distance_and_location_udf(col("Latitude"), col("Longitude"))
)

# Print the DataFrame with the new 'Distance' and 'Location' columns
filtered_df.select("Location").show(truncate=False)

####################################################################################################################
# Find the location where the maximum earthquake occurred
max_magnitude_row = filtered_df.orderBy(col("Magnitude").desc()).first()
print('\nLocation of Maximum Earthquake:', max_magnitude_row['Location'])

# Find the location where the minimum earthquake occurred
min_magnitude_row = filtered_df.orderBy(col("Magnitude")).first()
print('Location of Minimum Earthquake:', min_magnitude_row['Location'])

# Print the DataFrame with the new columns
filtered_df = filtered_df.withColumn("Latitude", col("Latitude").cast(DoubleType()))
filtered_df = filtered_df.withColumn("Longitude", col("Longitude").cast(DoubleType()))
filtered_df = filtered_df.withColumn("Depth", col("Depth").cast(DoubleType()))
filtered_df = filtered_df.withColumn("Magnitude", col("Magnitude").cast(DoubleType()))
filtered_df = filtered_df.withColumn("Depth Error", col("Depth Error").cast(DoubleType()))
filtered_df = filtered_df.withColumn("Magnitude Error", col("Magnitude Error").cast(DoubleType()))
filtered_df = filtered_df.withColumn("Azimuthal Gap", col("Azimuthal Gap").cast(DoubleType()))
filtered_df = filtered_df.withColumn("Horizontal Distance", col("Horizontal Distance").cast(DoubleType()))
filtered_df = filtered_df.withColumn("Horizontal Error", col("Horizontal Error").cast(DoubleType()))
filtered_df = filtered_df.withColumn("Root Mean Square", col("Root Mean Square").cast(DoubleType()))

# Create a new column 'Timestamp' using the 'Timestamp' column
filtered_df = filtered_df.withColumn("Timestamp", filtered_df["Timestamp"].cast("timestamp"))
filtered_df.cache()

# Save the final DataFrame to a CSV file
filtered_df.write.csv("/location/final_data.csv", header=True, mode="overwrite")
filtered_df.printSchema()


new_column_names = [col.replace(' ', '_') for col in filtered_df.columns]
filtered_df = filtered_df.toDF(*new_column_names)
    
filtered_df.printSchema()

# Function to create a folium map with earthquake markers
def create_map(data):
    # Create a base map centered around the reference location (0, 0)
    earthquake_map = folium.Map(location=[0, 0], zoom_start=2, tiles='OpenStreetMap')  # Use 'OpenStreetMap' tile

    # Create a MarkerCluster layer for better performance with a large number of markers
    marker_cluster = MarkerCluster().add_to(earthquake_map)

    # Convert PySpark DataFrame to a list of Row objects
    rows = data.collect()

    # Add markers for each earthquake location
    for row in rows:
        location_name = row['Location'] if 'Location' in data.columns else ''
        popup_content = f"<b>Location:</b> {location_name}<br><b>Magnitude:</b> {row['Magnitude']}<br><b>Depth:</b> {row['Depth']} km"
        tooltip_content = f"<b>Magnitude Level:</b> {row['Magnitude_Level']}"

        # You can customize the marker icon by using the 'icon' parameter
        folium.Marker([row['Latitude'], row['Longitude']],
                      popup=folium.Popup(html=popup_content, max_width=300),
                      tooltip=tooltip_content,
                      icon=folium.Icon(color='red', icon='info-sign')).add_to(marker_cluster)

    return earthquake_map

# Create a folium map with earthquake markers
earthquake_map = create_map(filtered_df[['Latitude', 'Longitude', 'Magnitude', 'Depth', 'Magnitude_Level', 'Location']])

# Save the map as an HTML fil
earthquake_map.save("/location/earthquake_map.html")

# Close the Spark session
spark.stop()


#spark-submit \
# --conf spark.network.timeout=300 \
# --conf spark.executor.heartbeatInterval=10\
# --conf spark.executor.memory=8g \
# --conf spark.executor.extraJavaOptions=-XX:-UseGCOverheadLimit \
# --driver-memory 4G \
# --executor-memory 8G \
# code.py
