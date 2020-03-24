package com.samuelberrien.phyvr.controls;

import android.content.SharedPreferences;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.widget.SeekBar;
import android.widget.TextView;

import com.samuelberrien.phyvr.R;

public class ControlActivity extends AppCompatActivity implements SharedPreferences.OnSharedPreferenceChangeListener {

	public static String controlSharedPref = "ControlSharedPref";

	@Override
	protected void onCreate(@Nullable Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		
		setContentView(R.layout.activity_control);

		SharedPreferences pref = getSharedPreferences(controlSharedPref, MODE_PRIVATE);

		SeekBar sk_dir = findViewById(R.id.seekbar_dir);
		TextView tv = findViewById(R.id.textview_dir);
		tv.setText("Direction ratio : " + pref.getFloat("Direction_ratio", 1.f));
		sk_dir.setProgress((int) (pref.getFloat("Direction_ratio", 1.f) * 100));
		sk_dir.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
			@Override
			public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
				float ratio = progress / 100.f;
				TextView tv = findViewById(R.id.textview_dir);
				tv.setText("Direction ratio : " + ratio);
				pref.edit().putFloat("Direction_ratio", ratio).apply();
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {

			}

			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {

			}
		});

		SeekBar sk_turret = findViewById(R.id.seekbar_turret);
		tv = findViewById(R.id.textview_turret);
		tv.setText("Turret ratio : " + pref.getFloat("Turret_ratio", 0.5f));
		sk_turret.setProgress((int) (pref.getFloat("Turret_ratio", 0.5f) * 100));
		sk_turret.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
			@Override
			public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
				float ratio = progress / 100.f;
				TextView tv = findViewById(R.id.textview_turret);
				tv.setText("Turret ratio : " + ratio);
				pref.edit().putFloat("Turret_ratio", ratio).apply();
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {

			}

			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {

			}
		});

		SeekBar sk_canon = findViewById(R.id.seekbar_canon);
		tv = findViewById(R.id.textview_canon);
		tv.setText("Canon ratio : " + pref.getFloat("Canon_ratio", 0.5f));
		sk_canon.setProgress((int) (pref.getFloat("Canon_ratio", 0.5f) * 100));
		sk_canon.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
			@Override
			public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
				float ratio = progress / 100.f;
				TextView tv = findViewById(R.id.textview_canon);
				tv.setText("Canon ratio : " + ratio);
				pref.edit().putFloat("Canon_ratio", ratio).apply();
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {

			}

			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {

			}
		});

		SeekBar sk_speed = findViewById(R.id.seekbar_speed);
		tv = findViewById(R.id.textview_speed);
		tv.setText("Speed ratio : " + pref.getFloat("Speed_ratio", 1.f));
		sk_speed.setProgress((int) (pref.getFloat("Speed_ratio", 1.f) * 100));
		sk_speed.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
			@Override
			public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
				float ratio = progress / 100.f;
				TextView tv = findViewById(R.id.textview_speed);
				tv.setText("Speed ratio : " + ratio);
				pref.edit().putFloat("Speed_ratio", ratio).apply();
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {

			}

			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {

			}
		});
	}

	@Override
	public void onSharedPreferenceChanged(SharedPreferences sharedPreferences, String key) {

	}
}
