import argostranslate.package

def setup_translation():
    print("Updating package index...")
    argostranslate.package.update_package_index()
    
    # Get available translation packages
    available_packages = argostranslate.package.get_available_packages()
    
    # List of Indian language codes (based on Argos Translate's codes)
    indian_languages = {"hi", "bn", "mr", "ta", "te", "gu", "kn", "pa", "ml", "ur", "or"}  # Add more if needed

    # Filter for only Indian language <-> English models
    filtered_packages = [
        pkg for pkg in available_packages
        if (pkg.from_code in indian_languages and pkg.to_code == "en") or 
           (pkg.from_code == "en" and pkg.to_code in indian_languages)
    ]

    print(f"Found {len(filtered_packages)} Indian language translation packages.")

    if not filtered_packages:
        print("No Indian language translation packages found.")
        return

    for package in filtered_packages:
        print(f"Installing {package.from_code} → {package.to_code}...")
        argostranslate.package.install_from_path(package.download())

    print("Translation setup complete!")

setup_translation()


