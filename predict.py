#this file must be copied to /Src/api
from flask import Blueprint, jsonify, request
predict_api = Blueprint('predict_api', __name__)

#@predict_api.route('/')
#def index():

@predict_api.route('predictor', methods=['POST'])
def apicall():
   return ('It works ;)')