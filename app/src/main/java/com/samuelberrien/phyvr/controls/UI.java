package com.samuelberrien.phyvr.controls;

import android.content.Context;

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

	boolean brake;
	boolean respawn;
	boolean fire;

	public UI(long levelPtr, JoyStick dirJoystick, Cursor speedCursor, JoyStick turretJoystick,
			  PlayButton fireButton, PlayButton brakeButton, PlayButton respawnButton) {
		this.controlPtr = levelPtr;
		dir = 0.f;
		turret = 0.f;
		canon = 0.f;
		speed = 0.f;

		brake = false;
		respawn = false;
		fire = false;

		dirJoystick.setJoyStickListener((float relX, float relY) -> {
				dir = relX;
		});

		speedCursor.setListener((float relY) -> {
				speed = -relY;
		});

		turretJoystick.setJoyStickListener((float relX, float relY) -> {
				turret = relX;
				canon = -relY;
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
