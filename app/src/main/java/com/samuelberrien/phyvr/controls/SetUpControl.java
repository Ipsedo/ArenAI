package com.samuelberrien.phyvr.controls;

import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.support.v4.content.ContextCompat;
import android.view.*;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;
import com.google.gson.Gson;
import com.samuelberrien.phyvr.R;

import java.util.Timer;
import java.util.TimerTask;

import static android.content.Context.MODE_PRIVATE;

public class SetUpControl extends LinearLayout {

	private Button setUpButton;
	private String controlName;
	private TextView controlView;

	private SharedPreferences controlsPref;

	private Controls.Infos infos;

	private boolean isSetUp;

	private boolean isActive;

	private long lastActivation;

	private StopColorationThread thread;

	public SetUpControl(final Activity activity, int controlStrId) {
		super(activity);
		setOrientation(HORIZONTAL);

		isActive = false;
		lastActivation = System.currentTimeMillis();

		controlName = activity.getString(controlStrId);

		controlsPref = activity.getSharedPreferences(Controls.sharedPref, MODE_PRIVATE);

		Gson gson = new Gson();
		String json = controlsPref.getString(controlName, Controls.Infos.getDefault());
		infos = gson.fromJson(json, Controls.Infos.class);

		LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT);
		params.weight = 0.5f;

		setUpButton = new Button(activity);
		setUpButton.setLayoutParams(params);
		setUpButton.setGravity(Gravity.CENTER);
		setUpButton.setText(controlName);

		setUpButton.setOnClickListener(new OnClickListener() {
			@Override
			public void onClick(View view) {
				isSetUp = true;
				controlView.setText("Searching...");

				Timer t = new Timer();
				t.schedule(new TimerTask() {
					@Override
					public void run() {
						isSetUp = false;
						activity.runOnUiThread(new Runnable() {
							@Override
							public void run() {
								setTextViewInfos();
							}
						});
					}
				}, 10000);
			}
		});

		addView(setUpButton);

		controlView = new TextView(activity);
		controlView.setLayoutParams(params);
		controlView.setGravity(Gravity.CENTER);
		setTextViewInfos();

		addView(controlView);

		thread = new StopColorationThread();
		thread.start();
	}

	private void setTextViewInfos() {
		controlView.setText(Integer.toString(infos.ID)
				+ (infos.isMotionEvent ? (infos.isPlusAxis ? "+" : "-") : ""));
	}

	public boolean motionEvent(MotionEvent event) {
		boolean handled = false;
		if (isSetUp) {
			for (int i = MotionEvent.AXIS_X; i < MotionEvent.AXIS_GENERIC_16; i++) {
				float value = event.getAxisValue(i);
				if (value > 0.5f || value < -0.5f) {
					infos.name = controlName;
					infos.isMotionEvent = true;
					infos.ID = i;
					infos.isPlusAxis = value > 0.f;

					SharedPreferences.Editor prefsEditor = controlsPref.edit();
					Gson gson = new Gson();
					String json = gson.toJson(infos);
					prefsEditor.putString(controlName, json);
					prefsEditor.apply();
					isSetUp = false;
					setTextViewInfos();
					handled = true;
					break;
				}
			}
		}
		float value = 0.f;
		if (infos.ID != -1)
			value = event.getAxisValue(infos.ID);
		if (infos.isMotionEvent && infos.isPlusAxis ? value > 0.1f : value < -0.1f) {
			isActive = true;
			setBackgroundColor(ContextCompat.getColor(getContext(), R.color.greyTransparent));
			handled = true;
		}
		if (handled)
			lastActivation = System.currentTimeMillis();
		return handled;
	}

	public boolean keyDown(int keyCode, KeyEvent event) {
		boolean handled = false;
		if (isSetUp && keyCode >= KeyEvent.KEYCODE_BUTTON_A && keyCode <= KeyEvent.KEYCODE_BUTTON_MODE) {
			infos.name = controlName;
			infos.isMotionEvent = false;
			infos.ID = keyCode;
			infos.isPlusAxis = false;

			SharedPreferences.Editor prefsEditor = controlsPref.edit();
			Gson gson = new Gson();
			String json = gson.toJson(infos);
			prefsEditor.putString(controlName, json);
			prefsEditor.apply();
			isSetUp = false;
			setTextViewInfos();
			handled = true;
		}
		if (infos.ID == keyCode) {
			isActive = true;
			setBackgroundColor(ContextCompat.getColor(getContext(), R.color.greyTransparent));
			handled = true;
		}
		if (handled)
			lastActivation = System.currentTimeMillis();
		return handled;
	}

	public void onResume() {
		thread = new StopColorationThread();
	}

	public void onPause() {
		thread.halt();
		try {
			thread.join();
		} catch (InterruptedException ie) {
			ie.printStackTrace();
		}
	}

	private class StopColorationThread extends Thread {
		private final long millisLimit;

		private boolean running;

		StopColorationThread() {
			super("StopColorationThread");
			running = true;
			millisLimit = 500;
		}

		void halt() {
			running = false;
		}

		@Override
		public void run() {
			while (running) {
				long curr = System.currentTimeMillis();
				if (isActive && curr - lastActivation > millisLimit) {
					isActive = false;
					setBackgroundColor(ContextCompat.getColor(getContext(), android.R.color.transparent));
				}
				try {
					Thread.sleep(500L);
				} catch (InterruptedException ie) {
					ie.printStackTrace();
				}
			}
		}
	}
}
