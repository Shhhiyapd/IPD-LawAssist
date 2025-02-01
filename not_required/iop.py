import pickle

with open(r'C:\Users\Shriya Deshpande\Downloads\index.pkl', 'rb') as file:
  data = pickle.load(file)

print(type(data))

if isinstance(data, dict):
  print(data.keys()) 
elif isinstance(data, list):
  print(data[:10]) # Show the first 10 items

if hasattr(data, '__dict__'):
  print(data.__dict__)

def explore_structure(data, indent=0):
  if isinstance(data, dict): 
    for key, value in data.items():
      print(' ' * indent + f'Key: {key}, Type: {type (value)}') 
      explore_structure (value, indent + 1)
  elif isinstance(data, list):
    print(' ' * indent + f'List of {len(data)} items')
    for item in data[:3]: # Show a few items for brevity
      explore_structure(item, indent + 1)
  else:
    print(' '* indent + f'Value: {data}, Type: {type(data)}')

explore_structure(data)