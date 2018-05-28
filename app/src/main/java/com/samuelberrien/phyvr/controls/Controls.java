package com.samuelberrien.phyvr.controls;

import android.view.InputDevice;
import android.view.MotionEvent;

import java.util.ArrayList;

public class Controls {

	private final long controlPtr;

	private int controllerId;

	static {
		System.loadLibrary("phyvr");
	}

	public native long initControls();

	public Controls(long controlPtr) {
		controllerId = -1;
		this.controlPtr = controlPtr;
	}

	public void onMotionEvent(MotionEvent motionEvent) {

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
}
