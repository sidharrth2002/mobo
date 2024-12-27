import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Example script")

# Add some arguments
parser.add_argument("--arg1", type=int, help="An integer argument", default=42)
parser.add_argument("--arg2", type=str, help="A string argument", default="Hello")
parser.add_argument("--flag", action=argparse.BooleanOptionalAction, help="A boolean flag")

# Parse the arguments
args = parser.parse_args()

# Convert the Namespace object to a dictionary
args_dict = vars(args)

# Print the dictionary
print(args_dict)
