package com.samuelberrien.phyvr.controls;

import android.content.Context;
import android.content.SharedPreferences;

import com.samuelberrien.phyvr.controls.ui.Cursor;
import com.samuelberrien.phyvr.controls.ui.JoyStick;
import com.samuelberrien.phyvr.controls.ui.PlayButton;

public class UI {

	static {
		System.loadLibrary("phyvr");
	}

	private final long controlPtr;

	private float dir;
	private float turret;
	private float canon;
	private float speed;

	private boolean brake;
	private boolean respawn;
	private boolean fire;

	public UI(Context context, long levelPtr, JoyStick dirJoystick, Cursor speedCursor, JoyStick turretJoystick,
			  PlayButton fireButton, PlayButton brakeButton, PlayButton respawnButton) {
		this.controlPtr = levelPtr;
		dir = 0.f;
		turret = 0.f;
		canon = 0.f;
		speed = 0.f;

		brake = false;
		respawn = false;
		fire = false;

		SharedPreferences controlPref = context.getSharedPreferences(ControlActivity.ControlSharedPref, Context.MODE_PRIVATE);

		dirJoystick.setJoyStickListener((float relX, float relY) -> {
				dir = relX * controlPref.getFloat(ControlActivity.DirectionRatioKey, 1.f);
		});

		speedCursor.setListener((float relY) -> {
				speed = -relY;
		});

		turretJoystick.setJoyStickListener((float relX, float relY) -> {
				turret = relX * controlPref.getFloat(ControlActivity.TurretRatioKey, 0.5f);
				canon = -relY * controlPref.getFloat(ControlActivity.CanonRatioKey, 0.5f);
		});

		fireButton.setListener((boolean clicked) -> {
			fire = clicked;
		});

		brakeButton.setListener((boolean clicked) -> {
			brake = clicked;
			speedCursor.reset();
		});

		respawnButton.setListener((boolean clicked) -> {
			respawn = clicked;
		});
	}

	public void sendInputs() {
		control(controlPtr, dir, speed, brake, turret, canon, respawn, fire);
	}

	public native void control(long levelPtr,
							   float direction,
							   float speed,
							   boolean brake,
							   float turretDir,
							   float turretUp,
							   boolean respawn,
							   boolean fire);
}
