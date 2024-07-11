"""
	Module: Mesage
	
	This script defines a node that represents a request with text and timestamp functionality.
	It includes methods to set the request text, set the current time, and copy the request text to the clipboard.
	
	Author: Torben Oschkinat - cgt104590 - Bachelor degree
	Date: 15.06.2024
"""

@tool
extends Node

const ERROR_COLOR = Color("#ff6b35")

# Reference nodes from the scene tree
@onready var request_text = $text
@onready var time_text = $time_text

var unformatted_text

# Function: _set_text
# Sets the text of the request.
# 
# @param text - The text content to be set for the request.
# @param is_typewriter - If the text should be displayed in typewriting style
func _set_text(text, is_typewriter):
	unformatted_text = text
	var formatted_text = text.replace("\t", "    ")
	var char_index = 0
	if is_typewriter:
		while char_index <= formatted_text.length():
			# Update the Label text
			request_text.text = formatted_text.substr(0, char_index)
			await get_tree().create_timer(0.05).timeout
			# Move to the next character
			char_index += 1
	else:
		request_text.text = formatted_text

func _set_error_color():
	request_text.modulate = ERROR_COLOR

# Function: _set_time
# Sets the current time as the request's timestamp.
func _set_time():
	time_text.text = _get_formated_time()

# Function: _get_formated_time
# Retrieves the current system time and formats it as a string in HH:MM format.
# 
# @return - A string representing the current time in HH:MM format.
func _get_formated_time():
	var time = Time.get_time_dict_from_system()
	return str(time.hour).pad_zeros(2) + ":" + str(time.minute).pad_zeros(2)

# Function: _on_copy_btn_pressed
# Copies the request text to the system clipboard when the copy button is pressed.
func _on_copy_btn_pressed():
	DisplayServer.clipboard_set(unformatted_text)
