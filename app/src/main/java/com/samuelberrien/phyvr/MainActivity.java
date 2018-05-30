package com.samuelberrien.phyvr;

import android.content.Intent;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);

		// Example of a call to a native method
		TextView tv = (TextView) findViewById(R.id.sample_text);
		//tv.setText(stringFromJNI());
		//tv.setText("YO");
		tv.setText(getString(R.string.app_name));
	}

	public void vr(View v) {
		Intent myIntent = new Intent(this, MyGvrActivity.class);
		startActivity(myIntent);
	}
}
