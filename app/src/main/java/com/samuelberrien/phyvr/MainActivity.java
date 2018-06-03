package com.samuelberrien.phyvr;

import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import com.samuelberrien.phyvr.controls.ControlActivity;
import com.samuelberrien.phyvr.normal.PlayActivity;
import com.samuelberrien.phyvr.vr.MyGvrActivity;

public class MainActivity extends AppCompatActivity {

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
	}

	public void normal(View v) {
		Intent myIntent = new Intent(this, PlayActivity.class);
		startActivity(myIntent);
	}

	public void vr(View v) {
		Intent myIntent = new Intent(this, MyGvrActivity.class);
		startActivity(myIntent);
	}

	public void configureControls(View v) {
		Intent myIntent = new Intent(this, ControlActivity.class);
		startActivity(myIntent);
	}
}
