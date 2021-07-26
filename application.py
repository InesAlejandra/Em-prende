##This file must be copied to /
from flask import Flask, jsonify, request, render_template, abort
# from Src.utils import ClassificationModelBuilder_short # Uncomment only if you want to re-train your model
from Src.api.predict import predict_api
from jinja2 import TemplateNotFound
#This next line is added to run in Colab
from flask_ngrok import run_with_ngrok

application = Flask(__name__ , template_folder='./Src/templates')
application.register_blueprint(predict_api, url_prefix='/em-prende-classification-model')
#Next line added to run in Colab
run_with_ngrok(application)


# Loading home page
@application.route('/', defaults={'page': 'index'})
@application.route('/<page>')
def show(page):

    try:
        print('home route')
        return render_template(f'{page}.html', app_name='Em-prende: Classification Problem')

    except TemplateNotFound:
        abort(404)


# Handling 400 Error
@application.errorhandler(400)
def bad_request(error=None):

    message = {
        'status': 400,
        'message': 'Bad Request: ' + request.url + '--> Please check your data payload...',
    }
    resp = jsonify(message)
    resp.status_code = 400
    
    return resp

# run application
if __name__ == "__main__":
    application.run()
