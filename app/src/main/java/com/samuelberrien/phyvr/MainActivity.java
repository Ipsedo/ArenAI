package com.samuelberrien.phyvr;

import android.content.Intent;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.TextView;
import com.google.gson.Gson;
import com.samuelberrien.phyvr.controls.ControlActivity;
import com.samuelberrien.phyvr.controls.Controls;

public class MainActivity extends AppCompatActivity {

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);

		Controls.Infos infos = new Controls.Infos();
		infos.ID = 1;
		infos.isMotionEvent = true;
		infos.name = "a";

		Gson gson = new Gson();
		String json = gson.toJson(infos);
		System.out.println(json);
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
