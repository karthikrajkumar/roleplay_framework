"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { GraduationCap, Settings, Mail, Lock, Building2 } from "lucide-react";
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
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-gray-900 to-slate-800 flex items-center justify-center p-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="w-full max-w-lg"
        >
          <Card className="backdrop-blur-xl bg-white/95 border-slate-200 shadow-2xl">
            <CardHeader className="text-center space-y-6 pb-8">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
                className="mx-auto w-16 h-16 bg-gradient-to-br from-slate-700 to-slate-900 rounded-full flex items-center justify-center shadow-lg border-2 border-slate-300"
              >
                <Building2 className="w-8 h-8 text-white" />
              </motion.div>
              
              <div>
                <CardTitle className="text-3xl font-bold text-slate-900 mb-3">
                  AI Coaching Platform
                </CardTitle>
                <CardDescription className="text-lg text-slate-600">
                  Transform your skills with AI-powered roleplay coaching
                </CardDescription>
              </div>
            </CardHeader>

            <CardContent className="space-y-8">
              <div className="text-center space-y-3">
                <h2 className="text-2xl font-semibold text-slate-900">Welcome Back</h2>
                <p className="text-slate-600">Choose your role and sign in to continue</p>
              </div>

              <div className="space-y-6">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <motion.div
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <Button
                      onClick={() => setSelectedRole('learner')}
                      className="w-full h-14 bg-gradient-to-r from-slate-700 to-slate-800 hover:from-slate-800 hover:to-slate-900 text-white font-semibold shadow-lg border border-slate-300"
                      size="lg"
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
                      variant="outline"
                      className="w-full h-14 border-2 border-slate-300 bg-white text-slate-700 hover:bg-slate-50 hover:border-slate-400 font-semibold shadow-lg"
                      size="lg"
                    >
                      <Settings className="w-5 h-5 mr-2" />
                      Content Admin
                    </Button>
                  </motion.div>
                </div>

                <Card className="bg-slate-50 border-slate-200 shadow-inner">
                  <CardContent className="text-center space-y-4 pt-6">
                    <div className="mx-auto w-12 h-12 bg-slate-100 rounded-full flex items-center justify-center border-2 border-slate-200">
                      <GraduationCap className="w-6 h-6 text-slate-600" />
                    </div>
                    <h3 className="font-semibold text-slate-900 text-lg">New to the platform?</h3>
                    <p className="text-slate-600">
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
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-gray-900 to-slate-800 flex items-center justify-center p-6">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3 }}
        className="w-full max-w-md"
      >
        <Card className="backdrop-blur-xl bg-white/95 border-slate-200 shadow-2xl">
          <CardHeader className="text-center space-y-4 pb-8">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: "spring", stiffness: 200 }}
              className="mx-auto w-16 h-16 bg-gradient-to-br from-slate-700 to-slate-900 rounded-full flex items-center justify-center shadow-lg border-2 border-slate-300"
            >
              {selectedRole === 'learner' ? (
                <GraduationCap className="w-8 h-8 text-white" />
              ) : (
                <Settings className="w-8 h-8 text-white" />
              )}
            </motion.div>
            
            <div>
              <CardTitle className="text-2xl font-bold text-slate-900 mb-2">
                Sign in as {selectedRole === 'learner' ? 'Learner' : 'Content Admin'}
              </CardTitle>
              <CardDescription className="text-slate-600">
                {selectedRole === 'learner' 
                  ? 'Access your assigned simulations and track progress'
                  : 'Manage content and learner assignments'
                }
              </CardDescription>
            </div>
          </CardHeader>

          <CardContent className="space-y-6">
            <form onSubmit={handleSignIn} className="space-y-5">
              <div className="space-y-2">
                <label className="text-sm font-medium text-slate-700">
                  Email Address
                </label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
                  <Input
                    type="email"
                    placeholder="Enter your email address"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="pl-10 h-12 bg-slate-50 border-slate-200 text-slate-900 placeholder:text-slate-400 focus:border-slate-400 focus:ring-slate-400/50"
                    required
                  />
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-slate-700">
                  Password
                </label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
                  <Input
                    type="password"
                    placeholder="Enter your password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="pl-10 h-12 bg-slate-50 border-slate-200 text-slate-900 placeholder:text-slate-400 focus:border-slate-400 focus:ring-slate-400/50"
                    required
                  />
                </div>
              </div>

              <Button
                type="submit"
                className="w-full h-12 bg-gradient-to-r from-slate-700 to-slate-800 hover:from-slate-800 hover:to-slate-900 text-white font-semibold shadow-lg mt-6"
              >
                Sign in as {selectedRole === 'learner' ? 'Learner' : 'Content Admin'}
              </Button>
            </form>

            <div className="text-center text-sm text-slate-600">
              Don&apos;t have an account?{' '}
              <button 
                onClick={resetSelection}
                className="text-slate-700 hover:text-slate-900 font-medium hover:underline"
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
          className="mt-6 text-center"
        >
          <Button
            variant="ghost"
            onClick={resetSelection}
            className="text-slate-300 hover:text-white hover:bg-white/10 font-medium"
          >
            ← Back to role selection
          </Button>
        </motion.div>
      </motion.div>
    </div>
  );
}