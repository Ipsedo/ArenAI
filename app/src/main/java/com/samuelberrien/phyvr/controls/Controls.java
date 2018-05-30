package com.samuelberrien.phyvr.controls;

import android.view.InputDevice;
import android.view.KeyEvent;
import android.view.MotionEvent;

import java.util.ArrayList;

public class Controls {

	private final long carPtr;

	private int controllerId;

	private int directionAxis;
	private int reverseAxis;
	private int speedAxis;

	static {
		System.loadLibrary("phyvr");
	}

	public native long initControls();

	public Controls(long carPtr) {
		controllerId = -1;

		this.carPtr = carPtr;

		/**
		 * Xbox one controller
		 */
		directionAxis = MotionEvent.AXIS_X;
		reverseAxis = MotionEvent.AXIS_BRAKE;
		speedAxis = MotionEvent.AXIS_THROTTLE;
	}

	public void onMotionEvent(MotionEvent event) {
		if ((event.getSource() & InputDevice.SOURCE_JOYSTICK)
				== InputDevice.SOURCE_JOYSTICK) {
			float dir = event.getAxisValue(directionAxis);
			float speed = event.getAxisValue(speedAxis);
			float reverse = event.getAxisValue(reverseAxis);

			float speedChange = reverse > 0.f ? reverse : speed;
			System.out.println("POIR : " + speedChange);
			control(carPtr, dir, speedChange);
		}
		if ((event.getSource() & InputDevice.SOURCE_DPAD)
				== InputDevice.SOURCE_DPAD) {
		}
	}

	public void onKeyDown(int keyCode, KeyEvent event) {
		switch(keyCode) {
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
		}
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
