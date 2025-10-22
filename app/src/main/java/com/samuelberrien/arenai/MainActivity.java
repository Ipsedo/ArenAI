package com.samuelberrien.arenai;

import android.app.NativeActivity;
import android.content.Intent;
import android.os.Bundle;

import android.view.View;
import android.widget.NumberPicker;
import android.widget.SeekBar;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.ActionBarDrawerToggle;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.view.WindowCompat;
import androidx.drawerlayout.widget.DrawerLayout;

import com.google.android.material.appbar.MaterialToolbar;
import com.google.android.material.navigation.NavigationView;
import com.samuelberrien.arenai.set_controls.ControlActivity;
import com.samuelberrien.arenai.set_controls.GamePadActivity;

public class MainActivity extends AppCompatActivity {

	private ActionBarDrawerToggle drawerToggle;

	int nb_tanks_chosen = 1;

    @Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);

		MaterialToolbar toolbar = findViewById(R.id.main_toolbar);
		setSupportActionBar(toolbar);

		DrawerLayout drawerLayout = findViewById(R.id.drawer_layout);
		drawerToggle = new ActionBarDrawerToggle(
				this, drawerLayout, toolbar,
				R.string.navigation_drawer_open, R.string.navigation_drawer_close
		);

		drawerLayout.addDrawerListener(drawerToggle);
		drawerToggle.syncState();

		TextView chosenEnemyNumberTextView = findViewById(R.id.chosen_enemy_number_textview);
		String originalChosenEnemyNumberMessage = chosenEnemyNumberTextView.getText().toString();
		chosenEnemyNumberTextView.setText(String.format(originalChosenEnemyNumberMessage, nb_tanks_chosen));

        SeekBar numberSeekBar = findViewById(R.id.enemy_number_seekbar);
		numberSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
			@Override
			public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
				nb_tanks_chosen = progress;
				chosenEnemyNumberTextView.setText(String.format(originalChosenEnemyNumberMessage, nb_tanks_chosen));
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
	public void onPostCreate(Bundle savedInstanceState) {
		super.onPostCreate(savedInstanceState);
		if (drawerToggle != null) drawerToggle.syncState();
	}

	@Override
	public void onConfigurationChanged(@NonNull android.content.res.Configuration newConfig) {
		super.onConfigurationChanged(newConfig);
		if (drawerToggle != null) drawerToggle.onConfigurationChanged(newConfig);
	}

	public void play(View v) {
		Intent intent = new Intent(this, NativeActivity.class);
		intent.putExtra("nb_tanks", nb_tanks_chosen);
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
