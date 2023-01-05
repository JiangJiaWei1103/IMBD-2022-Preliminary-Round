"""
Project-specific metadata for global access.
Author: Urban2.0
"""
# Primary key
PK = None  # Just use `df.index`
# Target column
TARGET = [f"sensor_point{p}_i_value" for p in range(5, 11)]
# Process category
PROC = ["clean", "oven", "painting", "env"]
