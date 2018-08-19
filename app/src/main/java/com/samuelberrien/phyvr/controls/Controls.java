package com.samuelberrien.phyvr.controls;

import android.content.Context;
import android.view.InputDevice;
import android.view.KeyEvent;
import android.view.MotionEvent;
import com.samuelberrien.phyvr.controls.axis.Axis;
import com.samuelberrien.phyvr.controls.button.Button;

import java.util.ArrayList;

public class Controls {


	private final long controlPtr;


	private float dir;
	private float turret;
	private float canon;
	private float speed;

	boolean brake;
	boolean respawn;
	boolean fire;

	private Axis dirAxis;
	private Axis speedAxis;
	private Axis turretAxis;
	private Axis canonAxis;

	private Button brakeButton;
	private Button respButton;
	private Button fireButton;

	static {
		System.loadLibrary("phyvr");
	}

	public Controls(Context context, long levelPtr) {
		this.controlPtr = levelPtr;
		dir = 0.f;
		turret = 0.f;
		canon = 0.f;
		speed = 0.f;

		brake = false;
		respawn = false;
		fire = false;

		// Axis
		dirAxis = new Axis(context, Axis.AxisMap.DIR);
		dirAxis.addListener((float value) -> {
					dir = value;
				}
		);

		speedAxis = new Axis(context, Axis.AxisMap.SPEED);
		speedAxis.addListener((float value) -> {
			speed = value;
		});

		turretAxis = new Axis(context, Axis.AxisMap.TURRET);
		turretAxis.addListener((float value) -> {
			turret = value;
		});

		canonAxis = new Axis(context, Axis.AxisMap.CANON);
		canonAxis.addListener((float value) -> {
			canon = value;
		});

		// Buttons
		brakeButton = new Button(context, Button.ButtonMap.BRAKE);
		brakeButton.addListener((boolean newState) -> {
			brake = newState;
		});

		respButton = new Button(context, Button.ButtonMap.RESPAWN);
		respButton.addListener((boolean newState) -> {
			respawn = newState;
		});

		fireButton = new Button(context, Button.ButtonMap.FIRE);
		fireButton.addListener((boolean newState) -> {
			fire = newState;
		});
	}

	public void sendInputs() {
		control(controlPtr, dir, speed, brake, turret, canon, respawn, fire);
	}

	public boolean onMotionEvent(MotionEvent event) {
		dirAxis.onGenericMotion(event);
		speedAxis.onGenericMotion(event);
		turretAxis.onGenericMotion(event);
		canonAxis.onGenericMotion(event);

		return true;
	}

	public boolean onKeyDown(int keyCode, KeyEvent event) {


		return brakeButton.onKeyDown(event) ||
				respButton.onKeyDown(event) ||
				fireButton.onKeyDown(event);
	}

	public boolean onKeyUp(int keyCode, KeyEvent event) {

		return brakeButton.onKeyUp(event) ||
				respButton.onKeyUp(event) ||
				fireButton.onKeyUp(event);
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
		int controllerId = gameControllerDeviceIds.get(0);
	}

	private class NoControllerException extends Exception {

	}

	public native void control(long levelPtr,
							   float direction,
							   float speed,
							   boolean brake,
							   float turretDir,
							   float turretUp,
							   boolean respawn,
							   boolean fire);

	public native void control2(long controlPtr, float[] arrayControl);
}
