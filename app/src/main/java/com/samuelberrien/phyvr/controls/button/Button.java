package com.samuelberrien.phyvr.controls.button;

import android.content.Context;
import android.content.SharedPreferences;
import android.view.KeyEvent;

import java.util.ArrayList;

public class Button implements SharedPreferences.OnSharedPreferenceChangeListener {

	public static String buttonPref = "buttonSharedPref";

	public enum ButtonMap {
		BRAKE(10, "BK", "Brake"),
		RESPAWN(11, "RSP", "Respawn"),
		FIRE(12, "FI", "Fire");

		private int id;
		private String name;
		private String fullName;

		ButtonMap(int id, String name, String fullName) {
			this.id = id;
			this.name = name;
			this.fullName = fullName;
		}

		public String getName() { return name; }
		public String getFullName() { return fullName; }
	}

	private int buttonID;
	private ButtonMap map;

	private boolean pressed;

	private ArrayList<OnStateChangedListener> list;

	public Button(Context context, ButtonMap map) {
		this.map = map;
		buttonID = -1;
		pressed = false;
		SharedPreferences pref = context.getSharedPreferences(buttonPref, Context.MODE_PRIVATE);
		pref.registerOnSharedPreferenceChangeListener(this);
		init(pref);
		list = new ArrayList<>();
	}

	private void init(SharedPreferences pref) {
		buttonID = pref.getInt(map.getName(), -1);
	}

	public ButtonMap getMap() {
		return map;
	}

	public int getID() {
		return buttonID;
	}

	public boolean onKeyDown(KeyEvent keyEvent) {
		if (keyEvent.getKeyCode() == buttonID) {
			for (OnStateChangedListener changed : list)
				changed.onStateChanged(true);
			return true;
		}
		return false;
	}

	public boolean onKeyUp(KeyEvent keyEvent) {
		if (keyEvent.getKeyCode() == buttonID) {
			for (OnStateChangedListener changed : list)
				changed.onStateChanged(false);
			return true;
		}
		return false;
	}

	public void addListener(OnStateChangedListener listener) {
		list.add(listener);
	}

	public void removeListener(OnStateChangedListener listener) {
		list.remove(listener);
	}

	@Override
	public void onSharedPreferenceChanged(SharedPreferences sharedPreferences, String s) {
		if (s.contains(map.getName()))
			init(sharedPreferences);
	}

	public interface OnStateChangedListener {
		void onStateChanged(boolean newState);
	}
}
