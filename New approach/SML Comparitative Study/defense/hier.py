import csv

# 1. Course Grades (Scale of 0 to 20)
with open('hierarchies/course_grade.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for i in range(21): # Covers 0 to 20
        # Portuguese grading: <10 is fail, 10-13 is sufficient, 14-17 is good, 18-20 is excellent
        level1 = "Fail (<10)" if i < 10 else "Pass (10-13)" if i <= 13 else "Good (14-17)" if i <= 17 else "Excellent (18-20)"
        writer.writerow([str(i), level1, "*"])

# 2. Unemployment Rate (Typically ranges from 5% to 20%)
with open('hierarchies/unemployment.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for i in range(50): # Cover 0 to 49
        level1 = "<10%" if i < 10 else "10-15%" if i <= 15 else ">15%"
        writer.writerow([str(i), level1, "*"])

# 3. Inflation Rate (Typically ranges from -5% to 10%)
with open('hierarchies/inflation.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for i in range(-20, 30): # Cover -20 to 29
        level1 = "Deflation (<0%)" if i < 0 else "Low (0-2%)" if i <= 2 else "High (>2%)"
        writer.writerow([str(i), level1, "*"])

# 4. GDP (Typically ranges from -10 to 10)
with open('hierarchies/gdp.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for i in range(-30, 30): # Cover -30 to 29
        level1 = "Recession (<0)" if i < 0 else "Growth (>=0)"
        writer.writerow([str(i), level1, "*"])

print("Final batch of hierarchies generated successfully!")