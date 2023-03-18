package com.samuelberrien.phyvr;

import android.app.NativeActivity;
import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.CheckBox;
import android.widget.NumberPicker;

import com.samuelberrien.phyvr.controls.ControlActivity;
import com.samuelberrien.phyvr.controls.GamePadActivity;
import com.samuelberrien.phyvr.game.PlayActivity;
import com.samuelberrien.phyvr.game.MyGvrActivity;

public class MainActivity extends AppCompatActivity implements NumberPicker.OnValueChangeListener {

	private int levelIdx;
	public static final String levelIdxExtraStr = "Level_Idx";
	public static final String useControllerExtraStr = "Use_Controller";

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);

		String[] levelName = new String[]{"Demo", "Practice"};
		NumberPicker levelPicker = findViewById(R.id.level_picker);
		levelPicker.setMinValue(0);
		levelPicker.setMaxValue(levelName.length - 1);
		levelPicker.setDisplayedValues(levelName);
		levelPicker.setValue(0);

		levelIdx = 0;

		levelPicker.setOnValueChangedListener(this);
	}

	public void normal(View v) {
		Intent myIntent = new Intent(this, PlayActivity.class);
		myIntent.putExtra(levelIdxExtraStr, levelIdx);
		myIntent.putExtra(useControllerExtraStr, ((CheckBox) findViewById(R.id.use_controller_switch)).isChecked());
		startActivity(myIntent);
	}

	public void vr(View v) {
		/*Intent myIntent = new Intent(this, MyGvrActivity.class);
		myIntent.putExtra(levelIdxExtraStr, levelIdx);
		startActivity(myIntent);*/
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

	@Override
	public void onValueChange(NumberPicker numberPicker, int oldVal, int newVal) {
		levelIdx = newVal;
	}
}
