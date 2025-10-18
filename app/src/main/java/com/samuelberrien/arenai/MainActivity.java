package com.samuelberrien.arenai;

import android.app.Activity;
import android.app.NativeActivity;
import android.content.Intent;
import android.os.Bundle;

import android.view.View;
import android.widget.NumberPicker;

import com.samuelberrien.arenai.set_controls.ControlActivity;
import com.samuelberrien.arenai.set_controls.GamePadActivity;

public class MainActivity extends Activity {

	public static final String useControllerExtraStr = "Use_Controller";

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
	}

	public void play(View v) {
		Intent intent = new Intent(this, NativeActivity.class);
		startActivity(intent);
	}

	public void configureGamePad(View v) {
		Intent myIntent = new Intent(this, GamePadActivity.class);
		startActivity(myIntent);
	}

	public void configureControls(View v) {
		Intent myIntent = new Intent(this, ControlActivity.class);
		startActivity(myIntent);
	}
}
