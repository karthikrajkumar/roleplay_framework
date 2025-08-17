"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { GraduationCap, Settings, Mail, Lock } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

type UserRole = 'learner' | 'admin' | null;

export default function HomePage() {
  const [selectedRole, setSelectedRole] = useState<UserRole>(null);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleSignIn = (e: React.FormEvent) => {
    e.preventDefault();
    console.log('Sign in:', { role: selectedRole, email, password });
  };

  const resetSelection = () => {
    setSelectedRole(null);
    setEmail("");
    setPassword("");
  };

  if (!selectedRole) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-gray-50 flex items-center justify-center p-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="w-full max-w-lg"
        >
          <Card className="bg-white border border-gray-200 shadow-lg shadow-gray-200/50 rounded-2xl">
            <CardHeader className="text-center space-y-4 pb-4 px-6 pt-6">
              <div>
                <CardTitle className="text-3xl font-bold text-gray-900 mb-2">
                  AI Coaching{" "}
                  <span className="bg-gradient-to-r from-blue-600 via-teal-600 to-green-600 bg-clip-text text-transparent">
                    Platform
                  </span>
                </CardTitle>
                <CardDescription className="text-base text-gray-600 max-w-md mx-auto leading-relaxed">
                  Transform your skills with AI-powered roleplay coaching, grounded in the methods you trust
                </CardDescription>
              </div>
            </CardHeader>

            <CardContent className="space-y-6 px-6 pb-6">
              <div className="text-center space-y-2">
                <h2 className="text-xl font-semibold text-gray-900">Welcome Back</h2>
                <p className="text-sm text-gray-500">Choose your role and sign in to continue</p>
              </div>

              <div className="space-y-4">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <motion.div
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <Button
                      onClick={() => setSelectedRole('learner')}
                      className="w-full h-12 bg-blue-50 hover:bg-blue-100 text-blue-700 border border-blue-200 hover:border-blue-300 font-semibold shadow-sm rounded-xl"
                    >
                      <GraduationCap className="w-5 h-5 mr-2" />
                      Learner
                    </Button>
                  </motion.div>

                  <motion.div
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <Button
                      onClick={() => setSelectedRole('admin')}
                      className="w-full h-12 bg-gray-50 hover:bg-gray-100 text-gray-700 border border-gray-200 hover:border-gray-300 font-semibold shadow-sm rounded-xl"
                    >
                      <Settings className="w-5 h-5 mr-2" />
                      Content Admin
                    </Button>
                  </motion.div>
                </div>

                <Card className="bg-gray-50 border border-gray-200 shadow-sm rounded-xl">
                  <CardContent className="text-center space-y-3 pt-4 pb-4">
                    <div className="mx-auto w-10 h-10 bg-gradient-to-br from-blue-100 to-green-100 rounded-xl flex items-center justify-center">
                      <GraduationCap className="w-5 h-5 text-gray-600" />
                    </div>
                    <h3 className="font-semibold text-gray-900 text-base">New to the platform?</h3>
                    <p className="text-gray-500 text-sm leading-relaxed">
                      Contact your administrator to get started with personalized AI coaching
                    </p>
                  </CardContent>
                </Card>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-gray-50 flex items-center justify-center p-6">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3 }}
        className="w-full max-w-md"
      >
        <Card className="bg-white border border-gray-200 shadow-lg shadow-gray-200/50 rounded-2xl">
          <CardHeader className="text-center space-y-4 pb-4 px-6 pt-6">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: "spring", stiffness: 200 }}
              className="mx-auto w-12 h-12 bg-gradient-to-br from-blue-500 via-teal-500 to-green-500 rounded-2xl flex items-center justify-center shadow-lg"
            >
              {selectedRole === 'learner' ? (
                <GraduationCap className="w-6 h-6 text-white" />
              ) : (
                <Settings className="w-6 h-6 text-white" />
              )}
            </motion.div>
            
            <div>
              <CardTitle className="text-xl font-bold text-gray-900 mb-1">
                Sign in as {selectedRole === 'learner' ? 'Learner' : 'Content Admin'}
              </CardTitle>
              <CardDescription className="text-sm text-gray-500 leading-relaxed">
                {selectedRole === 'learner' 
                  ? 'Access your assigned simulations and track progress'
                  : 'Manage content and learner assignments'
                }
              </CardDescription>
            </div>
          </CardHeader>

          <CardContent className="space-y-4 px-6 pb-6">
            <form onSubmit={handleSignIn} className="space-y-4">
              <div className="space-y-1">
                <label className="text-sm font-medium text-gray-700">
                  Email Address
                </label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <Input
                    type="email"
                    placeholder="Enter your email address"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="pl-10 h-10 bg-gray-50 border-gray-200 text-gray-900 placeholder:text-gray-400 focus:border-blue-500 focus:ring-blue-500/20 rounded-xl"
                    required
                  />
                </div>
              </div>

              <div className="space-y-1">
                <label className="text-sm font-medium text-gray-700">
                  Password
                </label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <Input
                    type="password"
                    placeholder="Enter your password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="pl-10 h-10 bg-gray-50 border-gray-200 text-gray-900 placeholder:text-gray-400 focus:border-blue-500 focus:ring-blue-500/20 rounded-xl"
                    required
                  />
                </div>
              </div>

              <Button
                type="submit"
                className="w-full h-10 bg-gray-100 hover:bg-gray-200 text-gray-800 border border-gray-200 hover:border-gray-300 font-semibold shadow-sm mt-4 rounded-xl"
              >
                Sign in as {selectedRole === 'learner' ? 'Learner' : 'Content Admin'}
              </Button>
            </form>

            <div className="text-center text-sm text-gray-500">
              Don&apos;t have an account?{' '}
              <button 
                onClick={resetSelection}
                className="text-gray-700 hover:text-gray-900 font-medium hover:underline"
              >
                Contact your administrator
              </button>
            </div>
          </CardContent>
        </Card>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="mt-4 text-center"
        >
          <Button
            variant="ghost"
            onClick={resetSelection}
            className="text-gray-500 hover:text-gray-700 hover:bg-gray-100 font-medium rounded-xl"
          >
            ← Back to role selection
          </Button>
        </motion.div>
      </motion.div>
    </div>
  );
}