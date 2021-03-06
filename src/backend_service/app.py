from flask import Flask, request, abort, make_response, jsonify
from flask_cors import CORS
from flask_marshmallow import Marshmallow

import database
import os

app = Flask(__name__)

# DB Setup
db = database.connect(app, 'postgres', os.environ['POSTGRES_PASSWORD'], 'postgres')
ma = Marshmallow(app)

# Cors Setup
CORS(app)

from controllers import conversationController, legalController


@app.route("/new", methods=['POST'])
def init_conversation():
    init_request = request.get_json()
    return conversationController.init_conversation(init_request['name'])


@app.route("/conversation", methods=['POST'])
def chat():
    chat_request = request.get_json()
    return conversationController.receive_message(chat_request['conversation_id'], chat_request['message'])


@app.route("/conversation/<conversation_id>", methods=['GET'])
def get_conversation(conversation_id=None):
    if conversation_id:
        return conversationController.get_conversation(conversation_id)
    else:
        abort(make_response(jsonify(message="Invalid request"), 400))


@app.route("/conversation/<conversation_id>/files", methods=['GET', 'POST'])
def handle_files(conversation_id=None):
    if conversation_id:
        if request.method == 'GET':
            return conversationController.get_file_list(conversation_id)

        if request.method == 'POST':
            if 'file' not in request.files:
                abort(make_response(jsonify(message="No file provided"), 400))

            return conversationController.upload_file(conversation_id, request.files['file'])
    else:
        abort(make_response(jsonify(message="Invalid request"), 400))


@app.route("/conversation/<conversation_id>/files/<file_id>", methods=['GET'])
def get_files(conversation_id=None, file_id=None):
    if conversation_id and file_id:
        return conversationController.get_file(conversation_id, file_id)
    else:
        abort(make_response(jsonify(message="Invalid request"), 400))


@app.route("/legal", methods=['GET'])
def get_legal_documents():
    return legalController.get_legal_documents()
