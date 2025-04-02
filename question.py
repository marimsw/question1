import pandas as pd
from transformers import pipeline

# Загрузка данных
data = pd.read_csv('freelancer_earnings.csv')

# Инициализация модели для ответов на вопросы
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def analyze_income_by_payment_method(method):
    """Сравнение доходов в зависимости от метода оплаты"""
    crypto_income = data[data['Payment_Method'] == 'Cryptocurrency']['Income'].mean()
    other_income = data[data['Payment_Method'] != 'Cryptocurrency']['Income'].mean()
    difference = crypto_income - other_income
    return crypto_income, other_income, difference

def income_distribution_by_region():
    """Распределение доходов по регионам"""
    return data.groupby('Region')['Income'].mean()

def percent_experts_with_less_than_100_projects():
    """Процент экспертов с менее 100 проектами"""
    experts = data[data['Expertise_Level'] == 'Expert']
    count_experts = experts.shape[0]
    less_than_100 = experts[experts['Projects_Completed'] < 100].shape[0]
    return (less_than_100 / count_experts) * 100 if count_experts > 0 else 0

def handle_query(query):
    """Обработка запросов"""
    if "криптовалюта" in query:
        crypto_income, other_income, difference = analyze_income_by_payment_method('Cryptocurrency')
        return (f"Средний доход фрилансеров, принимающих криптовалюту: {crypto_income:.2f}, "
                f"средний доход других: {other_income:.2f}, разница: {difference:.2f}")
    
    elif "регион" in query:
        distribution = income_distribution_by_region()
        return distribution.to_string()  # Показать распределение
    
    elif "менее 100 проектов" in query:
        percent = percent_experts_with_less_than_100_projects()
        return f"Процент экспертов, выполнивших менее 100 проектов: {percent:.2f}%"

    else:
        # Если запрос не соответствует известным шаблонам, используем модель для получения ответа
        context = data.to_string()  # Превращаем весь набор данных в текст для контекста
        result = qa_pipeline(question=query, context=context)
        return f"Ответ: {result['answer']}"

def main():
    while True:
        query = input("Введите ваш запрос: ")
        if query.lower() in ['выход', 'exit']:
            break
        response = handle_query(query)
        print(response)

if __name__ == '__main__':
    main()
