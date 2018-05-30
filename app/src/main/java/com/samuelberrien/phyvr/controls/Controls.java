package com.samuelberrien.phyvr.controls;

import android.content.Context;
import android.content.SharedPreferences;
import android.view.InputDevice;
import android.view.KeyEvent;
import android.view.MotionEvent;
import com.google.gson.Gson;
import com.samuelberrien.phyvr.R;

import java.util.ArrayList;

import static android.content.Context.MODE_PRIVATE;

public class Controls {

	public static class Infos {
		public int ID;
		public boolean isMotionEvent;
		public String name;
		public boolean isPlusAxis; // -1, +1 axis for motioneEvent
		public static String getDefault() {
			return "{\"ID\":-1,\"isMotionEvent\":false,\"isPlusAxis\":false,\"name\":\"undefined\"}";
		}
	}

	public static String sharedPref = "ControlsPreference";

	private final long carPtr;

	private SharedPreferences controlsPref;

	private int controllerId;

	private Infos leftInfo;
	private Infos rightInfo;
	private Infos speedUpInfo;
	private Infos speedDownInfo;

	static {
		System.loadLibrary("phyvr");
	}

	public native long initControls();

	public Controls(Context context, long carPtr) {
		controllerId = -1;

		this.carPtr = carPtr;

		controlsPref = context.getSharedPreferences(Controls.sharedPref, MODE_PRIVATE);
		Gson gson = new Gson();

		String json = controlsPref.getString(context.getString(R.string.left_control), Controls.Infos.getDefault());
		leftInfo = gson.fromJson(json, Controls.Infos.class);

		json = controlsPref.getString(context.getString(R.string.right_control), Controls.Infos.getDefault());
		rightInfo = gson.fromJson(json, Controls.Infos.class);

		json = controlsPref.getString(context.getString(R.string.speed_up_control), Controls.Infos.getDefault());
		speedUpInfo = gson.fromJson(json, Controls.Infos.class);

		json = controlsPref.getString(context.getString(R.string.speed_down_control), Controls.Infos.getDefault());
		speedDownInfo = gson.fromJson(json, Controls.Infos.class);
	}

	public void onMotionEvent(MotionEvent event) {
		float leftValue = 0.f;
		float rightValue = 0.f;
		float speedUpValue = 0.f;
		float speedDownValue = 0.f;
		if (leftInfo.isMotionEvent) {
			leftValue = event.getAxisValue(leftInfo.ID);
			leftValue = leftInfo.isPlusAxis ? -leftValue : leftValue;
		}
		if (rightInfo.isMotionEvent) {
			rightValue = event.getAxisValue(rightInfo.ID);
			rightValue = rightInfo.isPlusAxis ? rightValue : -rightValue;
		}
		if (speedUpInfo.isMotionEvent) {
			speedUpValue = event.getAxisValue(speedUpInfo.ID);
			speedUpValue = speedUpInfo.isPlusAxis ? speedUpValue : -speedUpValue;
		}
		if (speedDownInfo.isMotionEvent) {
			speedDownValue = event.getAxisValue(speedDownInfo.ID);
			speedDownValue = speedDownInfo.isPlusAxis ? -speedDownValue : speedDownValue;
		}

		float steer;
		float speed;
		if (Math.abs(leftValue) > Math.abs(rightValue)) steer = leftValue;
		else steer = rightValue;
		if (Math.abs(speedUpValue) > Math.abs(speedDownValue)) speed = speedUpValue;
		else speed = speedDownValue;

		control(carPtr, steer, speed);
	}

	public void onKeyDown(int keyCode, KeyEvent event) {
		/*switch(keyCode) {
			case KeyEvent.KEYCODE_BUTTON_L1:
			case KeyEvent.KEYCODE_BUTTON_R1:
			case KeyEvent.KEYCODE_BUTTON_THUMBR:
			case KeyEvent.KEYCODE_BUTTON_THUMBL:
			case KeyEvent.KEYCODE_DPAD_LEFT:
			case KeyEvent.KEYCODE_DPAD_RIGHT:
			case KeyEvent.KEYCODE_DPAD_UP:
			case KeyEvent.KEYCODE_DPAD_DOWN:
			case KeyEvent.KEYCODE_BUTTON_START:
			case KeyEvent.KEYCODE_BUTTON_MODE://Big button in the middle
			case KeyEvent.KEYCODE_BUTTON_B:
			case KeyEvent.KEYCODE_BUTTON_A:
			case KeyEvent.KEYCODE_BUTTON_X:
			case KeyEvent.KEYCODE_BUTTON_Y:
			default:
		}*/
	}

	public void onKeyUp(int keyCode, KeyEvent event) {

	}

	public void initControllerIds() throws NoControllerException {
		ArrayList<Integer> gameControllerDeviceIds = new ArrayList<>();
		int[] deviceIds = InputDevice.getDeviceIds();
		for (int deviceId : deviceIds) {
			InputDevice dev = InputDevice.getDevice(deviceId);
			int sources = dev.getSources();

			// Verify that the device has gamepad buttons, control sticks, or both.
			if (((sources & InputDevice.SOURCE_GAMEPAD) == InputDevice.SOURCE_GAMEPAD)
					|| ((sources & InputDevice.SOURCE_JOYSTICK)
					== InputDevice.SOURCE_JOYSTICK)) {
				// This device is a game controller. Store its device ID.
				if (!gameControllerDeviceIds.contains(deviceId)) {
					gameControllerDeviceIds.add(deviceId);
				}
			}
		}
		if (gameControllerDeviceIds.isEmpty()) {
			throw new NoControllerException();
		}
		controllerId = gameControllerDeviceIds.get(0);
	}

	private class NoControllerException extends Exception {

	}

	public native void control(long controlPtr, float direction, float speed);
}
