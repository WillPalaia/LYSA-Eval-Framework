import pytesseract
from PIL import Image
import pandas as pd
import re

# Set up tesseract executable path
# Note: Update the tesseract_cmd path to where Tesseract is installed on your system
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Sample test image path
image_path = 'download.png'

# Extract text from image
text = pytesseract.image_to_string(Image.open(image_path))

print(text)



# # Function to extract text from an image
# def extract_text_from_image(image_path):
#     image = Image.open(image_path)
#     text = pytesseract.image_to_string(image)
#     return text

# # Function to parse the text and extract names and ratings
# def parse_text(text):
#     lines = text.split('\n')
#     players_data = []

#     # Regex to match player data lines
#     pattern = re.compile(r'([A-Za-z\s-]+),(\d+),(\d+),(\d+),(\d+\.\d+),')
    
#     for line in lines:
#         match = pattern.match(line)
#         if match:
#             player_name = match.group(1).strip()
#             technical = int(match.group(2))
#             tactical = int(match.group(3))
#             effort = int(match.group(4))
#             overall = float(match.group(5))
#             players_data.append([player_name, technical, tactical, effort, overall])

#     return players_data

# # Function to create a DataFrame from the extracted data
# def create_dataframe(data):
#     df = pd.DataFrame(data, columns=['Player Name', 'Technical', 'Tactical', 'Effort', 'Overall'])
#     return df

# # Path to the image file
# image_path = 'download.png'  # Update this path

# # Extract text from image
# text = extract_text_from_image(image_path)

# # Parse the text to get player data
# player_data = parse_text(text)

# # Create DataFrame from the data
# df = create_dataframe(player_data)

# # Print the DataFrame
# print(df)

# # Save the DataFrame to a CSV file
# df.to_csv('player_ratings.csv', index=False)