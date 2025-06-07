import os
import pandas as pd
import random

# Set the directory where your images are stored (update the path if needed)
image_dir = 'C:/Users/manix/Downloads/compressed/food-101/food-101/images/Food'  # Update this path
output_csv = 'labels.csv'

# Get a list of all image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith(('jpg', 'jpeg', 'png'))]

# Create an empty list to hold the data
data = []

# Function to generate random nutritional values
def generate_nutrition():
    carbs = random.randint(20, 100)  # Random carbs between 20 and 100
    fats = random.randint(5, 50)     # Random fats between 5 and 50
    protein = random.randint(10, 60) # Random protein between 10 and 60
    return carbs, fats, protein

# Loop through all image files and generate random nutrition for each one
for image in image_files:
    carbs, fats, protein = generate_nutrition()
    data.append([image, carbs, fats, protein])

# Create a DataFrame from the data
df = pd.DataFrame(data, columns=['filename', 'carbs', 'fats', 'protein'])

# Save the DataFrame as a CSV
df.to_csv(output_csv, index=False)

print(f"✅ Labels CSV generated at {output_csv}")
