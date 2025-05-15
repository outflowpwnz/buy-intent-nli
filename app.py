from flask import Flask, request, jsonify
from transformers import pipeline

# Создание приложения Flask
app = Flask(__name__)

# Инициализация модели Hugging Face для zero-shot классификации
classifier = pipeline("zero-shot-classification", model="cointegrated/rubert-base-cased-nli-threeway")

# Эндпоинт для классификации
@app.route("/classify", methods=["POST"])
def classify_text():
    data = request.get_json()

    context = data.get("context", [])
    text = data.get("text", "")
    
    results = []
    
    for label in context:
        # Вызываем classifier с одним кандидатом в виде массива
        result = classifier(text, candidate_labels=[label])
        
        # Получаем оценку для этого единственного кандидата
        score = result["scores"][0] if result["scores"] else 0.0
        
        results.append({
            "label": label,
            "score": score
        })

    return jsonify(results)

# Запуск приложения
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
