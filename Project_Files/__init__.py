## remember to edit dash_packages imported

import dash
# codebase
from flask import Flask
server = Flask(__name__)

app = dash.Dash(__name__, server=server, url_base_pathname='/dashboard/')

app.config['suppress_callback_exceptions']=True

# from Project_Files.model import *
