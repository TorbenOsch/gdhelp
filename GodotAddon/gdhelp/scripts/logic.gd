"""
	Module: Chat logic
	
	This script defines the functionality for the chat interface.
	It handles sending requests and displaying responses.
	
	Author: Torben Oschkinat - cgt104590 - Bachelor degree
	Date: 15.06.2024
"""

@tool
extends Node

# Preload the request and answer templates
const REQUEST_TEMPLATE = preload("res://addons/gdhelp/presets/request.tscn")
const ANSWER_TEMPLATE = preload("res://addons/gdhelp/presets/answer.tscn")
const SERVER_IP = "127.0.0.1"

# Reference nodes from the scene tree
@onready var text_edit = $VBoxContainer/Input/TextEdit
@onready var container_requests = $VBoxContainer/ScrollContainer/Container_Requests
@onready var scroll_container = $VBoxContainer/ScrollContainer

# Variables
var curr_answer
var url = "http://" + SERVER_IP + ":8000"

# Function: _on_button_pressed
# This function is called when the send button is pressed.
# It creates a new request, scrolls the container to the bottom, and clears the text edit field.
func _on_button_pressed():
	if text_edit.text != '':
		# Add new request to the list
		_create_new_request(text_edit.text)
		# Set the scrollbar to the bottom
		scroll_container.scroll_vertical = scroll_container.get_v_scroll_bar().max_value
		# Reset the text edit
		text_edit.text = ''

# Function: _create_new_request
# This function creates a new request from the template, adds it to the container,
# and sets its text and timestamp. It also creates a placeholder answer.
#
# @param request - The text content of the request.
func _create_new_request(request):
	var curr_request = REQUEST_TEMPLATE.instantiate()
	container_requests.add_child(curr_request)
	curr_request._set_text(request, false)
	curr_request._set_time()
	
	# Creating the Request
	var http_request = HTTPRequest.new()
	add_child(http_request)
	http_request.request_completed.connect(_on_request_completed)
	var body = JSON.stringify(request)
	http_request.request(url, [], HTTPClient.METHOD_POST, body)
	curr_answer = ANSWER_TEMPLATE.instantiate()
	container_requests.add_child(curr_answer)
	curr_answer._set_time()
	curr_answer._set_text("Waiting for response...", true)

# Function: _on_request_completed
# This function handles the completion of an HTTP request.
# If the response code is 200, it sets the response text and updates the timestamp
# on the current answer object. If not, it sets an error message and adjusts the color
# of the current answer object to indicate an error state.
#
# @param result - The result of the HTTP request.
# @param response_code - The HTTP response code received.
# @param headers - The headers received in the response.
# @param body - The body of the HTTP response.
func _on_request_completed(result, response_code, headers, body):
	if response_code == 200:
		var response = body.get_string_from_utf8()
		curr_answer._set_text(response, true)
		curr_answer._set_time()
	else:
		curr_answer._set_text("An error accured during your request!", false)
		curr_answer._set_error_color()
