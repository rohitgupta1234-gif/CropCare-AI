"""
CropCare AI - Disease Information Database
Comprehensive disease information for all 16 PlantVillage classes.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from src.config import CLASS_NAMES, CROP_MAPPING, HEALTHY_CLASSES


@dataclass
class DiseaseInfo:
    """Disease information structure."""
    class_name: str
    crop: str
    disease_name: str
    is_healthy: bool
    symptoms: List[str]
    causes: List[str]
    prevention: List[str]
    treatment: List[str]
    severity: str  # Low, Medium, High
    spread_method: List[str]
    favorable_conditions: List[str]


# Disease information database (verified from PlantVillage and agricultural sources)
DISEASE_DATABASE: Dict[str, DiseaseInfo] = {
    # ==================== TOMATO ====================
    "Tomato_healthy": DiseaseInfo(
        class_name="Tomato_healthy",
        crop="Tomato",
        disease_name="Healthy",
        is_healthy=True,
        symptoms=["Green, vibrant leaves", "No visible lesions or discoloration", "Normal growth pattern"],
        causes=["Proper care and nutrition", "Disease-free environment", "Good agricultural practices"],
        prevention=[
            "Use certified disease-free seeds",
            "Practice crop rotation (3-4 years)",
            "Maintain proper plant spacing for air circulation",
            "Water at soil level, avoid overhead irrigation",
            "Remove plant debris after harvest",
            "Apply balanced fertilization",
        ],
        treatment=["No treatment needed - plant is healthy"],
        severity="None",
        spread_method=["N/A"],
        favorable_conditions=["N/A"],
    ),

    "Tomato_Bacterial_spot": DiseaseInfo(
        class_name="Tomato_Bacterial_spot",
        crop="Tomato",
        disease_name="Bacterial Spot",
        is_healthy=False,
        symptoms=[
            "Small, water-soaked lesions on leaves",
            "Lesions turn brown/black with yellow halos",
            "Spots on fruits - raised, scabby lesions",
            "Severe defoliation in advanced stages",
        ],
        causes=[
            "Xanthomonas campestris pv. vesicatoria (bacteria)",
            "Seed-borne infection",
            "Infected plant debris",
            "Contaminated tools and water",
        ],
        prevention=[
            "Use certified disease-free seeds",
            "Hot water seed treatment (50°C for 25 min)",
            "Crop rotation with non-host crops (2-3 years)",
            "Avoid overhead irrigation",
            "Sanitize tools between plants",
            "Remove and destroy infected plant debris",
        ],
        treatment=[
            "Copper-based bactericides (preventive)",
            "Streptomycin (where registered)",
            "Remove severely infected leaves",
            "Improve air circulation",
        ],
        severity="High",
        spread_method=["Water splash", "Wind-driven rain", "Contaminated tools", "Infected seeds"],
        favorable_conditions=["Warm temperatures (24-30°C)", "High humidity", "Wet foliage"],
    ),

    "Tomato_Early_blight": DiseaseInfo(
        class_name="Tomato_Early_blight",
        crop="Tomato",
        disease_name="Early Blight",
        is_healthy=False,
        symptoms=[
            "Concentric ring lesions (target spots) on lower leaves",
            "Dark brown spots with yellow halos",
            "Lesions enlarge and merge",
            "Stem lesions at soil line (collar rot)",
            "Fruit rot at stem end",
        ],
        causes=[
            "Alternaria solani (fungus)",
            "Overwinters in plant debris and soil",
            "Spores spread by wind and rain",
        ],
        prevention=[
            "Crop rotation (3-4 years)",
            "Stake plants to improve air flow",
            "Mulch to reduce soil splash",
            "Remove lower leaves as plants grow",
            "Avoid overhead watering",
            "Use resistant varieties when available",
        ],
        treatment=[
            "Fungicides: chlorothalonil, mancozeb, copper",
            "Apply at first sign, repeat 7-10 days",
            "Remove infected leaves",
            "Biofungicides: Bacillus subtilis",
        ],
        severity="Medium-High",
        spread_method=["Wind", "Rain splash", "Contaminated soil", "Tools"],
        favorable_conditions=["Warm (24-29°C)", "High humidity", "Alternating wet/dry"],
    ),

    "Tomato_Late_blight": DiseaseInfo(
        class_name="Tomato_Late_blight",
        crop="Tomato",
        disease_name="Late Blight",
        is_healthy=False,
        symptoms=[
            "Water-soaked, irregular greenish-black lesions",
            "White fuzzy growth on leaf undersides (sporulation)",
            "Rapid leaf death and defoliation",
            "Brown, firm lesions on fruits",
            "Entire plant can collapse in days",
        ],
        causes=[
            "Phytophthora infestans (oomycete)",
            "Survives in infected tubers/plant debris",
            "Airborne sporangia travel long distances",
        ],
        prevention=[
            "Plant resistant varieties",
            "Destroy volunteer potatoes/tomatoes",
            "Avoid overhead irrigation",
            "Wide row spacing for air circulation",
            "Monitor weather forecasts (Blightcast)",
            "Preventive fungicide schedule",
        ],
        treatment=[
            "Fungicides: chlorothalonil, mancozeb, copper (preventive)",
            "Systemic: mefenoxam, metalaxyl (resistance risk)",
            "Remove and destroy infected plants immediately",
            "Do not compost infected material",
        ],
        severity="Very High",
        spread_method=["Wind (long distance)", "Rain splash", "Infected tubers"],
        favorable_conditions=["Cool (15-21°C)", "High humidity (>90%)", "Wet foliage"],
    ),

    "Tomato_Leaf_Mold": DiseaseInfo(
        class_name="Tomato_Leaf_Mold",
        crop="Tomato",
        disease_name="Leaf Mold",
        is_healthy=False,
        symptoms=[
            "Yellow-green patches on upper leaf surface",
            "Olive-green to brown velvety growth on undersides",
            "Leaves curl, wither, and drop",
            "Usually starts on older leaves",
        ],
        causes=[
            "Passalora fulva (fungus, formerly Cladosporium)",
            "High humidity in greenhouses",
            "Poor air circulation",
        ],
        prevention=[
            "Ventilate greenhouses to reduce humidity",
            "Space plants adequately",
            "Avoid overhead watering",
            "Remove lower leaves",
            "Use resistant varieties",
        ],
        treatment=[
            "Fungicides: chlorothalonil, mancozeb",
            "Reduce humidity below 85%",
            "Increase air circulation",
            "Remove infected leaves",
        ],
        severity="Medium",
        spread_method=["Air currents", "Water splash", "Workers' clothing"],
        favorable_conditions=["High humidity (>85%)", "Moderate temps (18-24°C)", "Poor ventilation"],
    ),

    "Tomato_Septoria_leaf_spot": DiseaseInfo(
        class_name="Tomato_Septoria_leaf_spot",
        crop="Tomato",
        disease_name="Septoria Leaf Spot",
        is_healthy=False,
        symptoms=[
            "Numerous small circular spots (1-2mm)",
            "Dark brown margins with gray/tan centers",
            "Tiny black dots (pycnidia) in lesion centers",
            "Starts on lower leaves, moves upward",
            "Heavy defoliation, sunscald on fruits",
        ],
        causes=[
            "Septoria lycopersici (fungus)",
            "Overwinters in plant debris",
            "Splash-dispersed spores",
        ],
        prevention=[
            "Crop rotation (2-3 years)",
            "Remove plant debris",
            "Stake and prune for air flow",
            "Mulch to prevent soil splash",
            "Drip irrigation preferred",
        ],
        treatment=[
            "Fungicides: chlorothalonil, copper, mancozeb",
            "Apply at first sign, repeat 7-10 days",
            "Remove severely infected leaves",
        ],
        severity="Medium",
        spread_method=["Rain splash", "Wind", "Contaminated tools", "Workers"],
        favorable_conditions=["Warm (20-25°C)", "Wet weather", "High humidity"],
    ),

    "Tomato_Spider_mites_Two-spotted_spider_mite": DiseaseInfo(
        class_name="Tomato_Spider_mites_Two-spotted_spider_mite",
        crop="Tomato",
        disease_name="Two-spotted Spider Mite",
        is_healthy=False,
        symptoms=[
            "Fine stippling (white/yellow dots) on leaves",
            "Bronzing or silvering of leaves",
            "Fine webbing on leaf undersides",
            "Mites visible with hand lens (tiny, two dark spots)",
            "Leaf drop and plant stunting",
        ],
        causes=[
            "Tetranychus urticae (arachnid, not insect)",
            "Hot, dry conditions",
            "Dusty conditions favor mites",
            "Broad-spectrum insecticides kill predators",
        ],
        prevention=[
            "Maintain adequate humidity",
            "Avoid dust on plants",
            "Conserve natural predators (Phytoseiulus)",
            "Inspect transplants before planting",
            "Remove weeds (alternate hosts)",
        ],
        treatment=[
            "Miticides: abamectin, bifenazate, spiromesifen",
            "Insecticidal soaps and oils (contact only)",
            "Biological: predatory mites (Phytoseiulus persimilis)",
            "Strong water spray to dislodge mites",
            "Rotate miticide modes of action",
        ],
        severity="Medium-High",
        spread_method=["Wind", "Workers/clothing", "Contaminated plants"],
        favorable_conditions=["Hot (>30°C)", "Dry (<50% RH)", "Dusty"],
    ),

    "Tomato_Target_Spot": DiseaseInfo(
        class_name="Tomato_Target_Spot",
        crop="Tomato",
        disease_name="Target Spot",
        is_healthy=False,
        symptoms=[
            "Concentric rings (target-like) on leaves",
            "Dark brown centers with lighter rings",
            "Lesions on stems and petioles",
            "Fruit spots: sunken, dark with concentric rings",
            "Defoliation in severe cases",
        ],
        causes=[
            "Corynespora cassiicola (fungus)",
            "Survives in crop debris and soil",
            "Wide host range (cucumber, pepper, etc.)",
        ],
        prevention=[
            "Crop rotation (2+ years)",
            "Remove plant debris",
            "Avoid overhead irrigation",
            "Improve air circulation",
            "Resistant varieties if available",
        ],
        treatment=[
            "Fungicides: chlorothalonil, mancozeb, azoxystrobin",
            "Apply preventively in high-risk conditions",
            "Remove infected leaves",
        ],
        severity="Medium",
        spread_method=["Wind", "Rain splash", "Contaminated debris"],
        favorable_conditions=["Warm (25-30°C)", "High humidity", "Extended leaf wetness"],
    ),

    "Tomato_Tomato_YellowLeaf_Curl_Virus": DiseaseInfo(
        class_name="Tomato_Tomato_YellowLeaf_Curl_Virus",
        crop="Tomato",
        disease_name="Tomato Yellow Leaf Curl Virus (TYLCV)",
        is_healthy=False,
        symptoms=[
            "Upward curling and cupping of leaves",
            "Yellowing of leaf margins and between veins",
            "Stunted growth, reduced leaf size",
            "Flower drop, poor fruit set",
            "Plants appear bushy and erect",
        ],
        causes=[
            "Tomato yellow leaf curl virus (Begomovirus)",
            "Transmitted by whitefly (Bemisia tabaci)",
            "Not seed-borne, not mechanical",
            "Weed reservoirs (e.g., nightshade)",
        ],
        prevention=[
            "Use virus-free transplants",
            "Whitefly control: reflective mulch, insecticides",
            "Row covers early season",
            "Remove weeds and volunteer tomatoes",
            "Plant resistant/tolerant varieties",
            "Crop-free period to break cycle",
        ],
        treatment=[
            "No cure for infected plants - remove and destroy",
            "Control whitefly vectors aggressively",
            "Insecticides: imidacloprid, thiamethoxam, spiromesifen",
            "Roguing of symptomatic plants",
        ],
        severity="Very High",
        spread_method=["Whitefly vector (Bemisia tabaci)"],
        favorable_conditions=["Warm climates", "High whitefly populations", "Year-round host availability"],
    ),

    "Tomato_Tomato_mosaic_virus": DiseaseInfo(
        class_name="Tomato_Tomato_mosaic_virus",
        crop="Tomato",
        disease_name="Tomato Mosaic Virus (ToMV)",
        is_healthy=False,
        symptoms=[
            "Mosaic pattern: light/dark green patches",
            "Leaf distortion, fernleaf appearance",
            "Stunted growth",
            "Brown streaks on stems/petioles",
            "Fruit: uneven ripening, internal browning",
        ],
        causes=[
            "Tomato mosaic virus (Tobamovirus)",
            "Mechanical transmission (hands, tools, clothing)",
            "Seed-borne (seed coat contamination)",
            "Root contact in hydroponics",
            "Very stable virus (survives years in debris)",
        ],
        prevention=[
            "Use certified virus-free seed",
            "Seed treatment: trisodium phosphate or heat",
            "Strict sanitation: hand washing, tool disinfection",
            "No smoking (tobacco can carry TMV/ToMV)",
            "Remove infected plants immediately",
            "Resistant varieties (Tm-2² gene)",
        ],
        treatment=[
            "No chemical cure - prevention only",
            "Remove and destroy infected plants",
            "Disinfect tools (10% bleach, 20% nonfat milk)",
            "Crop rotation (2+ years)",
        ],
        severity="High",
        spread_method=["Mechanical contact", "Contaminated tools/hands", "Seed", "Root contact"],
        favorable_conditions=["Any - highly stable virus", "Dense plantings", "Frequent handling"],
    ),

    # ==================== POTATO ====================
    "Potato_healthy": DiseaseInfo(
        class_name="Potato_healthy",
        crop="Potato",
        disease_name="Healthy",
        is_healthy=True,
        symptoms=["Green, turgid leaves", "No lesions or discoloration", "Vigorous growth"],
        causes=["Good seed quality", "Proper rotation", "Adequate nutrition", "Disease management"],
        prevention=[
            "Use certified seed tubers",
            "Crop rotation (3-4 years)",
            "Proper hilling",
            "Balanced fertilization",
            "Timely irrigation",
        ],
        treatment=["No treatment needed"],
        severity="None",
        spread_method=["N/A"],
        favorable_conditions=["N/A"],
    ),

    "Potato_Early_blight": DiseaseInfo(
        class_name="Potato_Early_blight",
        crop="Potato",
        disease_name="Early Blight",
        is_healthy=False,
        symptoms=[
            "Concentric ring lesions on lower leaves",
            "Dark brown spots with yellow halos",
            "Target-like appearance",
            "Stem lesions (collar rot)",
            "Tuber lesions: dark, sunken, leathery",
        ],
        causes=[
            "Alternaria solani (fungus)",
            "Overwinters in debris and soil",
            "Stress predisposes plants",
        ],
        prevention=[
            "Crop rotation (3-4 years)",
            "Certified disease-free seed",
            "Adequate nitrogen (reduces stress)",
            "Avoid overhead irrigation",
            "Remove plant debris",
            "Resistant varieties",
        ],
        treatment=[
            "Fungicides: chlorothalonil, mancozeb, azoxystrobin",
            "Start at row closure or first sign",
            "Repeat 7-14 days",
            "Tank mix to prevent resistance",
        ],
        severity="Medium-High",
        spread_method=["Wind", "Rain splash", "Soil", "Equipment"],
        favorable_conditions=["Warm (24-29°C)", "High humidity", "Plant stress", "Poor nutrition"],
    ),

    "Potato_Late_blight": DiseaseInfo(
        class_name="Potato_Late_blight",
        crop="Potato",
        disease_name="Late Blight",
        is_healthy=False,
        symptoms=[
            "Water-soaked lesions on leaves/stems",
            "White sporulation on undersides (humid)",
            "Rapid necrosis, 'burnt' appearance",
            "Tuber rot: brown, granular, firm",
            "Secondary bacterial soft rot common",
        ],
        causes=[
            "Phytophthora infestans (oomycete)",
            "Infected seed tubers",
            "Cull piles, volunteers",
            "Long-distance airborne spread",
        ],
        prevention=[
            "Certified seed only",
            "Destroy cull piles and volunteers",
            "Hill well to protect tubers",
            "Forecast-based spray programs (Blightcast)",
            "Resistant varieties (e.g., Sarpo Mira)",
            "Preventive fungicide schedule",
        ],
        treatment=[
            "Protective: chlorothalonil, mancozeb, copper",
            "Curative/systemic: mefenoxam, cymoxanil, fluazinam",
            "Vine kill before harvest if severe",
            "Do not harvest wet",
        ],
        severity="Very High",
        spread_method=["Wind (sporangia)", "Rain", "Infected tubers"],
        favorable_conditions=["Cool (15-21°C)", "High humidity (>90%)", "Frequent rain/fog"],
    ),

    # ==================== PEPPER (BELL) ====================
    "Pepper_bell_healthy": DiseaseInfo(
        class_name="Pepper_bell_healthy",
        crop="Pepper",
        disease_name="Healthy",
        is_healthy=True,
        symptoms=["Dark green, glossy leaves", "No spots or wilting", "Strong fruit set"],
        causes=["Good cultural practices", "Disease-free seed/transplants", "Proper rotation"],
        prevention=[
            "Certified disease-free seed",
            "Crop rotation (2-3 years)",
            "Raised beds for drainage",
            "Drip irrigation",
            "Balanced fertility",
        ],
        treatment=["No treatment needed"],
        severity="None",
        spread_method=["N/A"],
        favorable_conditions=["N/A"],
    ),

    "Pepper_bell_Bacterial_spot": DiseaseInfo(
        class_name="Pepper_bell_Bacterial_spot",
        crop="Pepper",
        disease_name="Bacterial Spot",
        is_healthy=False,
        symptoms=[
            "Small water-soaked leaf spots",
            "Brown necrotic centers with yellow halos",
            "Spots coalesce, leaves yellow and drop",
            "Fruit: raised, scabby, brown lesions",
            "Severe defoliation, sunburn on fruit",
        ],
        causes=[
            "Xanthomonas campestris pv. vesicatoria (same as tomato)",
            "Seed-borne",
            "Infected transplants",
            "Rain/irrigation splash",
        ],
        prevention=[
            "Hot water seed treatment (50°C, 25 min)",
            "Disease-free transplants",
            "Crop rotation (2+ years)",
            "Avoid overhead irrigation",
            "Copper sprays preventive",
            "Sanitize greenhouse surfaces",
        ],
        treatment=[
            "Copper bactericides (preventive)",
            "Streptomycin (transplants, where allowed)",
            "Actigard (acibenzolar-S-methyl) - SAR inducer",
            "Remove infected leaves/fruit",
        ],
        severity="High",
        spread_method=["Water splash", "Wind-driven rain", "Tools", "Workers", "Seed"],
        favorable_conditions=["Warm (24-30°C)", "High humidity", "Wet foliage", "Wind-driven rain"],
    ),

    "Pepper_bell__Leaf_spot": DiseaseInfo(
        class_name="Pepper_bell__Leaf_spot",
        crop="Pepper",
        disease_name="Cercospora Leaf Spot (Frogeye Leaf Spot)",
        is_healthy=False,
        symptoms=[
            "Circular lesions with gray centers, dark borders",
            "Frogeye appearance (concentric rings)",
            "Lesions 3-10mm diameter",
            "Severe spotting causes defoliation",
            "Stem and petiole lesions",
            "Fruit spots: sunken, dark",
        ],
        causes=[
            "Cercospora capsici (fungus)",
            "Survives in crop debris",
            "Seed-borne",
            "Wind-dispersed conidia",
        ],
        prevention=[
            "Crop rotation (2-3 years)",
            "Disease-free seed",
            "Remove plant debris",
            "Avoid overhead irrigation",
            "Wide row spacing",
        ],
        treatment=[
            "Fungicides: chlorothalonil, azoxystrobin, trifloxystrobin",
            "Apply at first sign",
            "Resistant varieties if available",
        ],
        severity="Medium",
        spread_method=["Wind", "Rain splash", "Seed", "Debris"],
        favorable_conditions=["Warm (25-30°C)", "High humidity", "Extended leaf wetness"],
    ),
}


def get_disease_info(class_name: str) -> Optional[DiseaseInfo]:
    """Get disease information by class name."""
    return DISEASE_DATABASE.get(class_name)


def get_all_diseases() -> List[DiseaseInfo]:
    """Get all disease information."""
    return [DISEASE_DATABASE[name] for name in CLASS_NAMES if name in DISEASE_DATABASE]


def get_diseases_by_crop(crop: str) -> List[DiseaseInfo]:
    """Get diseases for a specific crop."""
    class_names = CROP_MAPPING.get(crop, [])
    return [DISEASE_DATABASE[name] for name in class_names if name in DISEASE_DATABASE]


def get_healthy_classes() -> List[DiseaseInfo]:
    """Get all healthy class information."""
    return [DISEASE_DATABASE[name] for name in HEALTHY_CLASSES if name in DISEASE_DATABASE]


def get_diseased_classes() -> List[DiseaseInfo]:
    """Get all diseased class information."""
    diseased = [name for name in CLASS_NAMES if name not in HEALTHY_CLASSES]
    return [DISEASE_DATABASE[name] for name in diseased if name in DISEASE_DATABASE]


def search_diseases(query: str) -> List[DiseaseInfo]:
    """Search diseases by name or symptoms."""
    query = query.lower()
    results = []
    for disease in DISEASE_DATABASE.values():
        if (query in disease.disease_name.lower() or
            query in disease.class_name.lower() or
            any(query in s.lower() for s in disease.symptoms)):
            results.append(disease)
    return results


# Export for easy access
__all__ = [
    "DiseaseInfo",
    "DISEASE_DATABASE",
    "get_disease_info",
    "get_all_diseases",
    "get_diseases_by_crop",
    "get_healthy_classes",
    "get_diseased_classes",
    "search_diseases",
]


if __name__ == "__main__":
    # Test
    print(f"Total diseases in database: {len(DISEASE_DATABASE)}")
    for name, info in DISEASE_DATABASE.items():
        print(f"  {name}: {info.disease_name} ({'Healthy' if info.is_healthy else 'Diseased'})")