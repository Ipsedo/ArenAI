package com.samuelberrien.phyvr.controls;

import android.content.SharedPreferences;
import android.os.Bundle;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import android.widget.SeekBar;
import android.widget.TextView;

import com.samuelberrien.phyvr.R;

public class ControlActivity extends AppCompatActivity implements SharedPreferences.OnSharedPreferenceChangeListener {

	public static String ControlSharedPref = "ControlSharedPref";

	public static String DirectionRatioKey = "Direction_ratio";
	public static String TurretRatioKey = "Turret_ratio";
	public static String CanonRatioKey = "Canon_ratio";
	public static String SpeedRatioKey = "Speed_ratio";

	@Override
	protected void onCreate(@Nullable Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);

		setContentView(R.layout.activity_control);

		SharedPreferences pref = getSharedPreferences(ControlSharedPref, MODE_PRIVATE);

		SeekBar sk_dir = findViewById(R.id.seekbar_dir);
		TextView tv = findViewById(R.id.textview_dir);
		tv.setText("ratio : " + pref.getFloat(DirectionRatioKey, 1.f));
		sk_dir.setProgress((int) (pref.getFloat(DirectionRatioKey, 1.f) * 100));
		sk_dir.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
			@Override
			public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
				float ratio = progress / 100.f;
				TextView tv = findViewById(R.id.textview_dir);
				tv.setText("ratio : " + ratio);
				pref.edit().putFloat(DirectionRatioKey, ratio).apply();
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {}

			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {}
		});

		SeekBar sk_turret = findViewById(R.id.seekbar_turret);
		tv = findViewById(R.id.textview_turret);
		tv.setText("ratio : " + pref.getFloat(TurretRatioKey, 0.5f));
		sk_turret.setProgress((int) (pref.getFloat(TurretRatioKey, 0.5f) * 100));
		sk_turret.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
			@Override
			public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
				float ratio = progress / 100.f;
				TextView tv = findViewById(R.id.textview_turret);
				tv.setText("ratio : " + ratio);
				pref.edit().putFloat(TurretRatioKey, ratio).apply();
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {}

			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {}
		});

		SeekBar sk_canon = findViewById(R.id.seekbar_canon);
		tv = findViewById(R.id.textview_canon);
		tv.setText("ratio : " + pref.getFloat(CanonRatioKey, 0.5f));
		sk_canon.setProgress((int) (pref.getFloat(CanonRatioKey, 0.5f) * 100));
		sk_canon.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
			@Override
			public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
				float ratio = progress / 100.f;
				TextView tv = findViewById(R.id.textview_canon);
				tv.setText("ratio : " + ratio);
				pref.edit().putFloat(CanonRatioKey, ratio).apply();
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {}

			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {}
		});

		SeekBar sk_speed = findViewById(R.id.seekbar_speed);
		tv = findViewById(R.id.textview_speed);
		tv.setText("ratio : " + pref.getFloat(SpeedRatioKey, 1.f));
		sk_speed.setProgress((int) (pref.getFloat(SpeedRatioKey, 1.f) * 100));
		sk_speed.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
			@Override
			public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
				float ratio = progress / 100.f;
				TextView tv = findViewById(R.id.textview_speed);
				tv.setText("ratio : " + ratio);
				pref.edit().putFloat(SpeedRatioKey, ratio).apply();
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {}

			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {}
		});
	}

	@Override
	public void onSharedPreferenceChanged(SharedPreferences sharedPreferences, String key) {

	}
}
