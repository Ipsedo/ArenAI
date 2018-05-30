package com.samuelberrien.phyvr.controls;

import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.view.*;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;
import com.google.gson.Gson;

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

	public SetUpControl(final Context context, int controlStrId, final Activity activity) {
		super(context);
		setOrientation(HORIZONTAL);

		controlName = context.getString(controlStrId);

		controlsPref = context.getSharedPreferences(Controls.sharedPref, MODE_PRIVATE);

		Gson gson = new Gson();
		String json = controlsPref.getString(controlName, Controls.Infos.getDefault());
		infos = gson.fromJson(json, Controls.Infos.class);

		LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT);
		params.weight = 0.5f;

		setUpButton = new Button(context);
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

		controlView = new TextView(context);
		controlView.setLayoutParams(params);
		controlView.setGravity(Gravity.CENTER);
		setTextViewInfos();

		addView(controlView);
	}

	private void setTextViewInfos() {
		controlView.setText(Integer.toString(infos.ID)
				+ (infos.isMotionEvent ? (infos.isPlusAxis ? "+" : "-") : ""));
	}

	public void motionEvent(MotionEvent event) {
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
					break;
				}
			}
		}
	}

	public void keyDown(int keyCode, KeyEvent event) {
		if (isSetUp) {
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
		}
	}
}
