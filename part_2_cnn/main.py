import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python main.py [train|evaluate]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "train":
        from src.train import main as train_main
        train_main()
    elif command == "evaluate":
        from src.evaluate import main as evaluate_main
        evaluate_main()
    else:
        print(f"Commande inconnue : {command}")