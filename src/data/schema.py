CLIENT_NAMES = ("changping", "baoding", "weifang")

ENVIRONMENTAL_CHANNELS = (
    "air_temperature",
    "relative_humidity",
    "illuminance",
    "atmospheric_pressure",
)

SOIL_CHANNELS = (
    "volumetric_water_content",
    "electrical_conductivity",
    "soil_temperature",
)

STRUCTURED_CHANNELS = (
    "historical_yield",
    "seed_cost",
    "fertilizer_cost",
    "labor_cost",
    "irrigation_cost",
)

SEASONS = (2022, 2023, 2024)
COMPARTMENTS_PER_CLIENT = 2
WINDOWS_PER_COMPARTMENT = 2
WINDOW_LENGTH = 336
IMAGE_SIZE = 224
INTERVAL_MINUTES = 30
DIURNAL_STEPS = 48
