package com.samuelberrien.arenai;

import android.app.NativeActivity;
import android.content.Intent;
import android.os.Bundle;

import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.SeekBar;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.ActionBarDrawerToggle;
import androidx.appcompat.app.AppCompatActivity;
import androidx.drawerlayout.widget.DrawerLayout;

import com.google.android.material.appbar.MaterialToolbar;
import com.samuelberrien.arenai.set_controls.ControlActivity;
import com.samuelberrien.arenai.set_controls.GamePadActivity;

import java.util.Map;

public class MainActivity extends AppCompatActivity {

    private ActionBarDrawerToggle drawerToggle;

    private int nbTanksChosen = 1;

    private Map<String, String> difficultyLevelToExecutorchModelAsset;
    private Spinner spinner;

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
        chosenEnemyNumberTextView.setText(String.format(originalChosenEnemyNumberMessage, nbTanksChosen));

        SeekBar numberSeekBar = findViewById(R.id.enemy_number_seekbar);
        numberSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                nbTanksChosen = progress;
                chosenEnemyNumberTextView.setText(String.format(originalChosenEnemyNumberMessage, nbTanksChosen));
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
            }
        });

        // enemies level
        spinner = findViewById(R.id.enemy_level_spinner);

        String[] levels = {"Easy", "Medium", "Hard"};

        difficultyLevelToExecutorchModelAsset = Map.of(
                levels[0], "executorch/actor.pte",
                levels[1], "executorch/actor.pte",
                levels[2], "executorch/actor.pte"
        );

        ArrayAdapter<String> adapter = new ArrayAdapter<>(
                this,
                R.layout.spinner_item,
                levels
        );
        adapter.setDropDownViewResource(R.layout.spinner_dropdown_item);

        spinner.setAdapter(adapter);
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
        intent.putExtra("nb_tanks", nbTanksChosen);
        intent.putExtra("executorch_model_asset", difficultyLevelToExecutorchModelAsset.get(spinner.getSelectedItem().toString()));
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
