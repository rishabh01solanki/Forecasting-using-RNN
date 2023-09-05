import random
import csv

# Define the number of data points
num_data_points = 1000

# Initialize an empty list to store the data
data = []

# Constants for the linear relationship between size and price
# These are derived from real-world approximations.
a = 233.33  # Coefficient for size
b = 0      # Base price, assuming no other factors

# Generate random data points with a more realistic relationship
for _ in range(num_data_points):
    size = random.randint(500, 30000)  # Random size between 500 and 30000 ft^2
    
    # Generate price based on size with a more realistic linear relationship
    # Add some random noise to the price to make it more realistic
    noise = random.uniform(-100000, 100000)
    price = (a * size) + b + noise
    
    # Convert the price to units of $1000 for consistency with the original code
    price = price / 1000
    
    data.append((size, price))

# Define the CSV file name
csv_filename = "realistic_synthetic_dataset.csv"

# Write the data to a CSV file
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    # Write the header row
    csv_writer.writerow(["Size (ft^2)", "Price ($1000)"])
    
    # Write the data rows
    csv_writer.writerows(data)

print(f"{num_data_points} data points with a more realistic relationship have been generated and saved to '{csv_filename}'.")
