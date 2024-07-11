"""
	Module: gdhelp
	
	This script defines an Editor Plugin for the Godot Engine that adds a dockable help panel 
	to the editor. The dock is loaded from a scene file and added to the left bottom dock slot 
	of the editor when the plugin is activated, and removed when the plugin is deactivated.
	
	Author: Torben Oschkinat - cgt104590 - Bachelor degree
	Date: 15.06.2024
"""

@tool
extends EditorPlugin

# Reference to the dock control
var dock

# Function: _enter_tree
# This function is called when the plugin is activated.
# It instantiates the dock scene and adds it to the left bottom dock slot of the editor.
func _enter_tree():
	dock = preload("res://addons/gdhelp/GdHelp.tscn").instantiate()
	add_control_to_dock(DOCK_SLOT_LEFT_BL, dock)

# Function: _exit_tree
# This function is called when the plugin is deactivated.
# It removes the dock control from the editor and frees its resources.
func _exit_tree():
	remove_control_from_docks(dock)
	dock.free()
