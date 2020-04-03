package com.samuelberrien.phyvr.controls;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Build;
import android.os.VibrationEffect;
import android.os.Vibrator;

import com.samuelberrien.phyvr.controls.ui.Cursor;
import com.samuelberrien.phyvr.controls.ui.JoyStick;
import com.samuelberrien.phyvr.controls.ui.PlayButton;

public class UI {

	static {
		System.loadLibrary("phyvr");
	}

	private Context context;

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

		this.context = context;

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
		if (vibrate(controlPtr)) {
			Vibrator v = (Vibrator) context.getSystemService(Context.VIBRATOR_SERVICE);
			if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
				v.vibrate(VibrationEffect.createOneShot(50, VibrationEffect.DEFAULT_AMPLITUDE));
			} else {
				//deprecated in API 26
				v.vibrate(50);
			}
		}
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

	public native boolean vibrate(long level_ptr);
}
