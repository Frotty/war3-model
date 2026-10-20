pipeline {
    agent { dockerfile { filename 'ci/Dockerfile'; args '-u root --entrypoint=""' } }
    options {
        timestamps()
        timeout(time: 30, unit: 'MINUTES')
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '30', artifactDaysToKeepStr: '14'))
    }
    triggers { cron(env.BRANCH_NAME == 'master' ? '20 21 * * 5' : '') }
    stages {
        stage('Dependencies') {
            steps {
                sh '''
                    set -eu
                    git config --global --add safe.directory "$WORKSPACE"
                    npm ci
                    npm install --no-save --package-lock=false @microsoft/eslint-formatter-sarif@2.1.7
                '''
            }
        }
        stage('TypeScript') { steps { sh 'npm run typecheck' } }
        stage('ESLint analysis') {
            steps {
                script {
                    int status = sh(returnStatus: true, script: 'npx eslint . --ext .ts --format @microsoft/eslint-formatter-sarif --output-file eslint-results.sarif')
                    if (status > 1) { error("ESLint could not run (exit ${status})") }
                    if (status == 1) { echo 'ESLint found issues; see the SARIF report and GitHub code scanning.' }
                }
            }
        }
        stage('Upload code scanning results') {
            when { allOf { not { changeRequest() }; branch 'master' } }
            steps {
                withCredentials([usernamePassword(credentialsId: 'token',
                        usernameVariable: 'GITHUB_ACTOR', passwordVariable: 'GITHUB_TOKEN')]) {
                    sh 'python3 ci/upload-sarif.py'
                }
            }
        }
    }
    post {
        always { archiveArtifacts artifacts: 'eslint-results.sarif', allowEmptyArchive: true }
        cleanup { sh 'chmod -R a+rwX "$WORKSPACE" 2>/dev/null || true' }
    }
}
