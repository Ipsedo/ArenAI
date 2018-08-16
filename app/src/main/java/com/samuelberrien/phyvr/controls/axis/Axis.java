package com.samuelberrien.phyvr.controls.axis;

import android.content.Context;
import android.content.SharedPreferences;
import android.view.MotionEvent;
import android.view.View;
import com.samuelberrien.phyvr.R;

import java.util.ArrayList;

public class Axis implements SharedPreferences.OnSharedPreferenceChangeListener {

	public static String axisPref = "AxisSharedPref";

	public enum AxisMap {
		DIR(0, "DIR", "Direction"),
		SPEED(1, "SPEED", "Speed"),
		CANON(2, "CANON", "Canon"),
		TURRET(3, "TURRET", "Turret");

		private int id;
		private String name;
		private String fullName;

		AxisMap(int id, String name, String fullName){
			this.id = id;
			this.name = name;
			this.fullName = fullName;
		}

		public int getValue() { return id; }

		public String getName() { return name; }

		public String getFullName() { return fullName; }
	}

	public static float LIMIT = 1e-1f;

	private int idPlus;
	private int idMinus;

	private float value;

	private AxisMap axisMap;

	private ArrayList<OnAxisMoveListener> listeners;

	private Context context;

	Axis(Context context, AxisMap axisMap) {
		idMinus = -1;
		idPlus = -1;
		value = 0.f;
		this.context = context;
		this.axisMap = axisMap;
		this.listeners = new ArrayList<>();
		SharedPreferences pref = context.getSharedPreferences(axisPref, Context.MODE_PRIVATE);
		pref.registerOnSharedPreferenceChangeListener(this);
		init(pref);

	}

	public AxisMap getAxisMap() {
		return axisMap;
	}

	public void addListener(OnAxisMoveListener listener) {
		listeners.add(listener);
	}

	public void removeListener(OnAxisMoveListener listener) {
		listeners.remove(listener);
	}

	private void init(SharedPreferences pref) {
		idMinus = pref.getInt(axisMap.getName() + "-", -1);
		idPlus = pref.getInt(axisMap.getName() + "+", -1);
	}

	public boolean onGenericMotion(MotionEvent event) {
		value = 0.f;

		if (event.getAxisValue(idPlus) > LIMIT) {
			value = Math.abs(event.getAxisValue(idPlus));
		}
		if (event.getAxisValue(idMinus) < -LIMIT) {
			value = -Math.abs(event.getAxisValue(idMinus));
		}

		for (OnAxisMoveListener l : listeners)
			l.valueChanged(value);

		return false;
	}

	@Override
	public void onSharedPreferenceChanged(SharedPreferences sharedPreferences, String key) {
		if (key.contains(axisMap.getName()))
			init(sharedPreferences);
	}

	public interface OnAxisMoveListener {
		void valueChanged(float value);
	}
}
