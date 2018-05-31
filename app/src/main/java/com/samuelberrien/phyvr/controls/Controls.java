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

	private final static float AXIS_LIMIT = 0.1f;

	private final long controlPtr;

	private SharedPreferences controlsPref;

	private int controllerId;

	private Infos leftInfo;
	private Infos rightInfo;
	private Infos speedUpInfo;
	private Infos speedDownInfo;
	private Infos brakeInfo;
	private Infos turretLeftInfo;
	private Infos turretRightInfo;
	private Infos turretUpInfo;
	private Infos turretDownInfo;
	private Infos respawnInfo;
	private Infos fireInfo;

	static {
		System.loadLibrary("phyvr");
	}

	public Controls(Context context, long controlPtr) {
		controllerId = -1;

		this.controlPtr = controlPtr;

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

		json = controlsPref.getString(context.getString(R.string.brake_control), Controls.Infos.getDefault());
		brakeInfo = gson.fromJson(json, Controls.Infos.class);

		json = controlsPref.getString(context.getString(R.string.turret_left_control), Controls.Infos.getDefault());
		turretLeftInfo = gson.fromJson(json, Controls.Infos.class);

		json = controlsPref.getString(context.getString(R.string.turret_right_control), Controls.Infos.getDefault());
		turretRightInfo = gson.fromJson(json, Controls.Infos.class);

		json = controlsPref.getString(context.getString(R.string.turret_up_control), Controls.Infos.getDefault());
		turretUpInfo = gson.fromJson(json, Controls.Infos.class);

		json = controlsPref.getString(context.getString(R.string.turret_down_control), Controls.Infos.getDefault());
		turretDownInfo = gson.fromJson(json, Controls.Infos.class);

		json = controlsPref.getString(context.getString(R.string.respawn_control), Controls.Infos.getDefault());
		respawnInfo = gson.fromJson(json, Controls.Infos.class);

		json = controlsPref.getString(context.getString(R.string.fire_control), Controls.Infos.getDefault());
		fireInfo = gson.fromJson(json, Controls.Infos.class);
	}

	public void onMotionEvent(MotionEvent event) {
		float leftValue = 0.f;
		float rightValue = 0.f;

		float speedUpValue = 0.f;
		float speedDownValue = 0.f;

		boolean brake = false;

		float leftTurret = 0.f;
		float rightTurret = 0.f;

		float upTurret = 0.f;
		float downTurret = 0.f;

		boolean respawn = false;

		boolean fire = false;

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

		if (brakeInfo.isMotionEvent)
			if (event.getAxisValue(brakeInfo.ID) > 0.5f) brake = true;

		if (turretLeftInfo.isMotionEvent) {
			leftTurret = event.getAxisValue(turretLeftInfo.ID);
			leftTurret = turretLeftInfo.isPlusAxis ? leftTurret : -leftTurret;
		}
		if (turretRightInfo.isMotionEvent) {
			rightTurret = event.getAxisValue(turretRightInfo.ID);
			rightTurret = turretRightInfo.isPlusAxis ? -rightTurret : rightTurret;
		}

		if (turretUpInfo.isMotionEvent) {
			upTurret = event.getAxisValue(turretUpInfo.ID);
			upTurret = turretUpInfo.isPlusAxis ? -upTurret : upTurret;
		}
		if (turretDownInfo.isMotionEvent) {
			downTurret = event.getAxisValue(turretDownInfo.ID);
			downTurret = turretDownInfo.isPlusAxis ? downTurret : -downTurret;
		}

		if (respawnInfo.isMotionEvent)
			if (event.getAxisValue(respawnInfo.ID) > 0.5f) respawn = true;

		if (fireInfo.isMotionEvent)
			if (event.getAxisValue(fireInfo.ID) > 0.5f) fire = true;

		float steer;
		float speed;
		float turretDir;
		float turretHeight;

		if (Math.abs(leftValue) > Math.abs(rightValue)) steer = leftValue;
		else steer = rightValue;

		if (Math.abs(speedUpValue) > Math.abs(speedDownValue)) speed = speedUpValue;
		else speed = speedDownValue;

		if (Math.abs(leftTurret) > Math.abs(rightTurret)) turretDir = leftTurret;
		else turretDir = rightTurret;

		if (Math.abs(upTurret) > Math.abs(downTurret)) turretHeight = upTurret;
		else turretHeight = downTurret;

		steer = Math.abs(steer) > AXIS_LIMIT ? steer : 0.f;
		speed = Math.abs(speed) > AXIS_LIMIT ? speed : 0.f;
		turretDir = Math.abs(turretDir) > AXIS_LIMIT ? turretDir : 0.f;
		turretHeight = Math.abs(turretHeight) > AXIS_LIMIT ? turretHeight : 0.f;

		control(controlPtr, steer, speed, brake, turretDir, turretHeight, respawn, fire);
	}

	public void onKeyDown(int keyCode, KeyEvent event) {
		float leftValue = 0.f;
		float rightValue = 0.f;

		float speedUpValue = 0.f;
		float speedDownValue = 0.f;

		boolean brake = false;

		float leftTurret = 0.f;
		float rightTurret = 0.f;

		float upTurret = 0.f;
		float downTurret = 0.f;

		boolean respawn = false;

		boolean fire = false;

		if (!leftInfo.isMotionEvent && leftInfo.ID == keyCode) leftValue = -1.f;
		if (!rightInfo.isMotionEvent && rightInfo.ID == keyCode) rightValue = 1.f;
		if (!speedUpInfo.isMotionEvent && speedUpInfo.ID == keyCode) speedUpValue = 1.f;
		if (!speedDownInfo.isMotionEvent && speedDownInfo.ID == keyCode) speedDownValue = -1.f;
		if (!brakeInfo.isMotionEvent && brakeInfo.ID == keyCode) brake = true;
		if (!turretLeftInfo.isMotionEvent && turretLeftInfo.ID == keyCode) leftTurret = -1.f;
		if (!turretRightInfo.isMotionEvent && turretRightInfo.ID == keyCode) rightTurret = 1.f;
		if (!turretUpInfo.isMotionEvent && turretUpInfo.ID == keyCode) upTurret = 1.f;
		if (!turretDownInfo.isMotionEvent && turretDownInfo.ID == keyCode) downTurret = -1.f;
		if (!respawnInfo.isMotionEvent && respawnInfo.ID == keyCode) respawn = true;
		if (!fireInfo.isMotionEvent && fireInfo.ID == keyCode) fire = true;

		float steer = leftValue + rightValue;
		float speed = speedUpValue + speedDownValue;
		float turretDir = leftTurret + rightTurret;
		float turretHeight = upTurret + downTurret;

		control(controlPtr, steer, speed, brake, turretDir, turretHeight, respawn, fire);
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

	public native void control(long controlPtr,
							   float direction,
							   float speed,
							   boolean brake,
							   float turretDir,
							   float turretUp,
							   boolean respawn,
							   boolean fire);
}
