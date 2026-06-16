from src.infer import predict

result = predict('data/data/chest_xray/test/NORMAL/IM-0001-0001.jpeg')
print(f"Prédiction: {result['prediction']}")
print(f"Confiance NORMAL: {result['NORMAL']:.2%}")
print(f"Confiance PNEUMONIA: {result['PNEUMONIA']:.2%}")