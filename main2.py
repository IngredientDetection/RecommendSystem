from flask import Flask, render_template, request
import based_Ingredient
app = Flask(__name__)

# 재료 목록 (예시로 몇 가지 재료를 포함합니다)
ingredients = ["치즈", "고기", "계란", "양파", "올리브", "베이컨", "파","버섯"]


@app.route('/recommend', methods=['POST'])
def recommend(selected_ingredients):
    input = ', '.join(selected_ingredients)
    rec = based_Ingredient.get_recs(input)
    print(rec)
    return render_template('button_select.html', results=rec)

@app.route('/')
def index():
    return render_template('button_select.html', ingredients=ingredients)

def dataframe_to_html(dataframe):
    return dataframe.to_html(classes='table table-striped', index=False)

@app.route('/select_ingredients', methods=['POST'])
def select_ingredients():
    selected_ingredients = request.form.getlist('ingredient')
    print(selected_ingredients)
    input = ', '.join(selected_ingredients)
    rec = based_Ingredient.get_recs(input)
    # 이 예시에서는 간단히 선택된 재료를 출력합니다.
    print(rec)
    html_table = dataframe_to_html(rec)\
    #item_list = rec.columns.values.tolist() + rec.values.tolist()
    return render_template('button_select.html',table=html_table)


if __name__ == '__main__':
    app.run(debug=True)