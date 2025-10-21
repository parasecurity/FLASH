def evaluate_fl_framework():
    print("FL Framework Evaluation Tool")
    print("============================")
    print("This tool evaluates your federated learning framework based on the criteria from")
    print("'Comparative analysis of open-source federated learning frameworks - a literature-based survey and review'")
    print("Answer the following questions about your FL framework to generate a comparable score.\n")

    framework_name = input("Enter your FL framework name: ")

    scores = {
        "features": {},
        "interoperability": {},
        "user_friendliness": {}
    }

    # Features Category (30% of total)
    print("\n--- FEATURES CATEGORY (30% of total score) ---")

    # Security Mechanisms (35% of Features)
    print("\nSecurity Mechanisms (35% of Features):")
    crypto = input(
        "Does your framework provide cryptographic security methods (e.g., homomorphic encryption, secure multiparty computation)? (y/n): ").lower() == 'y'
    algo = input(
        "Does your framework provide algorithmic security methods (e.g., differential privacy)? (y/n): ").lower() == 'y'

    if crypto and algo:
        scores["features"]["security"] = 1.0
    elif crypto or algo:
        scores["features"]["security"] = 0.5
    else:
        scores["features"]["security"] = 0.0

    # FL Algorithms (25% of Features)
    print("\nFL Algorithms (25% of Features):")
    fedavg = input("Does your framework provide FedAvg algorithm out of the box? (y/n): ").lower() == 'y'
    adaptive = input(
        "Does your framework provide adaptive optimization algorithms (e.g., FedProx, FedOpt, FedAdam) or asynchronous algorithms? (y/n): ").lower() == 'y'

    if fedavg and adaptive:
        scores["features"]["algorithms"] = 1.0
    elif fedavg:
        scores["features"]["algorithms"] = 0.5
    else:
        scores["features"]["algorithms"] = 0.0

    # ML Models (25% of Features)
    print("\nML Models (25% of Features):")
    multiple_libs = input(
        "Does your framework support multiple ML libraries (e.g., both TensorFlow and PyTorch)? (y/n): ").lower() == 'y'
    one_lib = input("Does your framework support at least one ML library? (y/n): ").lower() == 'y'

    if multiple_libs:
        scores["features"]["models"] = 1.0
    elif one_lib:
        scores["features"]["models"] = 0.5
    else:
        scores["features"]["models"] = 0.0

    # FL Paradigms (15% of Features)
    print("\nFL Paradigms (15% of Features):")
    horizontal = input("Does your framework support horizontal FL? (y/n): ").lower() == 'y'
    vertical = input("Does your framework support vertical FL? (y/n): ").lower() == 'y'

    if horizontal and vertical:
        scores["features"]["paradigms"] = 1.0
    elif horizontal:
        scores["features"]["paradigms"] = 0.0
    else:
        scores["features"]["paradigms"] = 0.0

    # Interoperability Category (20% of total)
    print("\n--- INTEROPERABILITY CATEGORY (20% of total score) ---")

    # Rollout To Edge Devices (50% of Interoperability)
    print("\nRollout To Edge Devices (50% of Interoperability):")
    full_rollout = input(
        "Does your framework support full rollout to edge devices without limitations? (y/n): ").lower() == 'y'
    limited_rollout = input(
        "Does your framework support rollout to edge devices with restrictions? (y/n): ").lower() == 'y'

    if full_rollout:
        scores["interoperability"]["rollout"] = 1.0
    elif limited_rollout:
        scores["interoperability"]["rollout"] = 0.5
    else:
        scores["interoperability"]["rollout"] = 0.0

    # OS Support (25% of Interoperability)
    print("\nOS Support (25% of Interoperability):")
    windows = input("Does your framework natively support Windows? (y/n): ").lower() == 'y'
    macos = input("Does your framework natively support MacOS? (y/n): ").lower() == 'y'

    if windows and macos:
        scores["interoperability"]["os_support"] = 1.0
    elif windows or macos:
        scores["interoperability"]["os_support"] = 0.5
    else:
        scores["interoperability"]["os_support"] = 0.0

    # GPU Support (15% of Interoperability)
    print("\nGPU Support (15% of Interoperability):")
    gpu = input("Does your framework support GPU acceleration? (y/n): ").lower() == 'y'

    scores["interoperability"]["gpu"] = 1.0 if gpu else 0.0

    # Docker Installation (10% of Interoperability)
    print("\nDocker Installation (10% of Interoperability):")
    docker = input("Is your framework available as a Docker container? (y/n): ").lower() == 'y'

    scores["interoperability"]["docker"] = 1.0 if docker else 0.0

    # User Friendliness Category (50% of total)
    print("\n--- USER FRIENDLINESS CATEGORY (50% of total score) ---")

    # Development Effort (25% of User Friendliness)
    print("\nDevelopment Effort (25% of User Friendliness):")
    print(
        "Based on your experiments or judgment, how would you rate the development effort required for your framework?")
    dev_effort = input("Low effort (1), Medium effort (0.5), High effort (0): ").lower()

    if dev_effort in ["1", "low"]:
        scores["user_friendliness"]["dev_effort"] = 1.0
    elif dev_effort in ["0.5", "medium"]:
        scores["user_friendliness"]["dev_effort"] = 0.5
    else:
        scores["user_friendliness"]["dev_effort"] = 0.0

    # Model Accuracy (25% of User Friendliness)
    print("\nModel Accuracy (25% of User Friendliness):")
    print("Based on your experiments (e.g., MNIST classification):")
    accuracy = input("Above 90% (1), Between 50-90% (0.5), Below 50% (0): ").lower()

    if accuracy in ["1", "above 90%"]:
        scores["user_friendliness"]["accuracy"] = 1.0
    elif accuracy in ["0.5", "between 50-90%"]:
        scores["user_friendliness"]["accuracy"] = 0.5
    else:
        scores["user_friendliness"]["accuracy"] = 0.0

    # Documentation (20% of User Friendliness)
    print("\nDocumentation (20% of User Friendliness):")
    print("How would you rate the documentation available for your framework?")
    docs = input("Extensive (1), Moderate (0.5), Minimal (0): ").lower()

    if docs in ["1", "extensive"]:
        scores["user_friendliness"]["documentation"] = 1.0
    elif docs in ["0.5", "moderate"]:
        scores["user_friendliness"]["documentation"] = 0.5
    else:
        scores["user_friendliness"]["documentation"] = 0.0

    # Training Speed (10% of User Friendliness)
    print("\nTraining Speed (10% of User Friendliness):")
    print("Based on your experiments (e.g., MNIST training time):")
    speed = input("Under 1 minute (1), Between 1-3 minutes (0.5), Over 3 minutes (0): ").lower()

    if speed in ["1", "under 1 minute"]:
        scores["user_friendliness"]["speed"] = 1.0
    elif speed in ["0.5", "between 1-3 minutes"]:
        scores["user_friendliness"]["speed"] = 0.5
    else:
        scores["user_friendliness"]["speed"] = 0.0

    # Data Preparation Effort (10% of User Friendliness)
    print("\nData Preparation Effort (10% of User Friendliness):")
    print("How would you rate the effort required to prepare data for your framework?")
    data_prep = input("Low effort (1), Medium effort (0.5), High effort (0): ").lower()

    if data_prep in ["1", "low"]:
        scores["user_friendliness"]["data_prep"] = 1.0
    elif data_prep in ["0.5", "medium"]:
        scores["user_friendliness"]["data_prep"] = 0.5
    else:
        scores["user_friendliness"]["data_prep"] = 0.0

    # Model Evaluation (5% of User Friendliness)
    print("\nModel Evaluation (5% of User Friendliness):")
    built_in = input("Does your framework provide built-in model evaluation methods? (y/n): ").lower() == 'y'
    difficult = input("Are there difficulties in implementing these evaluation methods? (y/n): ").lower() == 'y'

    if built_in and not difficult:
        scores["user_friendliness"]["evaluation"] = 1.0
    elif built_in and difficult:
        scores["user_friendliness"]["evaluation"] = 0.5
    else:
        scores["user_friendliness"]["evaluation"] = 0.0

    # Pricing Systems (5% of User Friendliness)
    print("\nPricing Systems (5% of User Friendliness):")
    free = input("Are all features of your framework freely available (not behind a paywall)? (y/n): ").lower() == 'y'

    scores["user_friendliness"]["pricing"] = 1.0 if free else 0.0

    # Calculate category scores
    features_score = (
                             scores["features"]["security"] * 0.35 +
                             scores["features"]["algorithms"] * 0.25 +
                             scores["features"]["models"] * 0.25 +
                             scores["features"]["paradigms"] * 0.15
                     ) * 100

    interop_score = (
                            scores["interoperability"]["rollout"] * 0.50 +
                            scores["interoperability"]["os_support"] * 0.25 +
                            scores["interoperability"]["gpu"] * 0.15 +
                            scores["interoperability"]["docker"] * 0.10
                    ) * 100

    user_friendly_score = (
                                  scores["user_friendliness"]["dev_effort"] * 0.25 +
                                  scores["user_friendliness"]["accuracy"] * 0.25 +
                                  scores["user_friendliness"]["documentation"] * 0.20 +
                                  scores["user_friendliness"]["speed"] * 0.10 +
                                  scores["user_friendliness"]["data_prep"] * 0.10 +
                                  scores["user_friendliness"]["evaluation"] * 0.05 +
                                  scores["user_friendliness"]["pricing"] * 0.05
                          ) * 100

    # Calculate total score - FIXED CALCULATION
    total_score = (
            features_score * 0.30 +
            interop_score * 0.20 +
            user_friendly_score * 0.50
    )

    # Display results
    print("\n--- EVALUATION RESULTS ---")
    print(f"Framework Name: {framework_name}")
    print(f"Features Score: {features_score:.2f}%")
    print(f"Interoperability Score: {interop_score:.2f}%")
    print(f"User Friendliness Score: {user_friendly_score:.2f}%")
    print(f"Total Score: {total_score:.2f}%")

    print("\nFor comparison with other frameworks from the paper:")
    print("Flower: 84.75%")
    print("FLARE: 80.50%")
    print("FederatedScope: 78.75%")
    print("PySyft: 72.50%")
    print("FedML: 71.00%")
    print("OpenFL: 69.00%")
    print("EasyFL: 67.50%")

    # Generate table data
    print("\n--- TABLE DATA  ---")
    print(f"Framework,Features Score,Interoperability Score,User Friendliness Score,Total Score")
    print(f"{framework_name},{features_score:.2f}%,{interop_score:.2f}%,{user_friendly_score:.2f}%,{total_score:.2f}%")


if __name__ == "__main__":
    evaluate_fl_framework()