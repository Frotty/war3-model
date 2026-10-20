pipeline {
    agent { dockerfile { filename 'ci/Dockerfile'; args '-u root --entrypoint=""' } }
    options {
        timestamps()
        timeout(time: 30, unit: 'MINUTES')
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '30', artifactDaysToKeepStr: '14'))
    }
    triggers { cron(env.BRANCH_NAME == 'master' ? '20 21 * * 5' : '') }
    environment { SARIF_GENERATED = 'false' }
    stages {
        stage('Dependencies') {
            steps {
                sh '''
                    set -eu
                    rm -f eslint-results.sarif
                    git config --global --add safe.directory "$WORKSPACE"
                    git clean -fdx -q
                    npm ci
                    npm install --no-save --package-lock=false @microsoft/eslint-formatter-sarif@2.1.7
                '''
            }
        }
        stage('TypeScript') {
            steps {
                // The former Actions jobs ran independently: keep analysis running
                // when type checking fails, while retaining a failing build result.
                catchError(buildResult: 'FAILURE', stageResult: 'FAILURE') {
                    sh 'npm run typecheck'
                }
            }
        }
        stage('ESLint analysis') {
            steps {
                script {
                    int status = sh(returnStatus: true, script: 'npx eslint . --ext .ts --format @microsoft/eslint-formatter-sarif --output-file eslint-results.sarif')
                    if (status > 1) { error("ESLint could not run (exit ${status})") }
                    if (!fileExists('eslint-results.sarif')) { error('ESLint did not produce its report') }
                    env.SARIF_GENERATED = 'true'
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
        always {
            script {
                if (env.SARIF_GENERATED == 'true') {
                    archiveArtifacts artifacts: 'eslint-results.sarif', allowEmptyArchive: false
                }
            }
        }
        cleanup { sh 'chmod -R a+rwX "$WORKSPACE" 2>/dev/null || true' }
    }
}
