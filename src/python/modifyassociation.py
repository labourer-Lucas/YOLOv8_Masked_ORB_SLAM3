

with open('associations.txt', 'r') as file:
    lines = file.readlines()

with open('modified_fassociations.txt', 'w') as file:
    for line in lines:
        columns = line.split()
        first_column = columns[0]
        fifth_column = f"mask/{first_column}.png"
        modified_line = f"{line.strip()} {fifth_column}\n"
        file.write(modified_line)