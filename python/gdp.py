import numpy as np
import matplotlib.pyplot as plt

# Constants
current_gdp_per_capita = 4300  # Vietnam's GDP per capita in USD (2024)
developed_threshold = 15000     # Estimated threshold for a developed nation
growth_rate = 0.06              # Annual GDP per capita growth rate (6%)

# Simulation
years = np.arange(2024, 2050)  # From 2024 to 2100
gdp_per_capita = [current_gdp_per_capita]

for year in years[1:]:
    new_gdp = gdp_per_capita[-1] * (1 + growth_rate)
    gdp_per_capita.append(new_gdp)

# Find the year Vietnam becomes a developed nation
year_developed = years[np.argmax(np.array(gdp_per_capita) >= developed_threshold)]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(years, gdp_per_capita, label="Vietnam GDP per Capita", marker='o', linestyle='-')
plt.axhline(y=developed_threshold, color='r', linestyle='--', label="Developed Nation Threshold")
plt.axvline(x=year_developed, color='g', linestyle='--', label=f"Developed by {year_developed}")

plt.xlabel("Year")
plt.ylabel("GDP per Capita (USD)")
plt.title("Vietnam GDP Per Capita Growth Projection")
plt.legend()
plt.grid(True)
plt.show()

# Output result
print(f"Vietnam is projected to become a developed nation by the year {year_developed}.")
